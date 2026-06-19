import torch
import torch.nn as nn
import numpy as np

class NeuralPointMap(nn.Module):
    # Constants for hash table state
    EMPTY_KEY = -1
    DELETED_KEY = -2

    def __init__(self, point_cloud_path=None, voxel_size=0.01, feature_dim=16, knn_k=10,
                 hash_table_size=2000000, num_nei_cells=1, search_alpha=1.0,
                 max_candidates=32, query_temperature=0.05, extra_probe_margin=20):
        super().__init__()
        self.knn_k = knn_k
        self.voxel_size = voxel_size
        self.hash_table_size = hash_table_size
        self.feature_dim = feature_dim
        self.max_candidates = max_candidates
        self.query_temperature = query_temperature
        self.extra_probe_margin = extra_probe_margin

        # Warn if knn_k exceeds max_candidates
        if knn_k > max_candidates:
            import warnings
            warnings.warn(
                f"knn_k ({knn_k}) exceeds max_candidates ({max_candidates}). "
                f"KNN results will be truncated to {max_candidates} neighbors."
            )

        # Primes for PIN-SLAM style hashing
        self.register_buffer("primes", torch.tensor([73856093, 19349669, 83492791], dtype=torch.int64))

        if point_cloud_path is not None:
            raw_points = torch.from_numpy(np.load(point_cloud_path)).float()
        else:
            raw_points = torch.empty((0, 3), dtype=torch.float32, device=self._device)

        # Build hash map without collisions at initialization
        self._initialize_map(raw_points, feature_dim)
        self.set_search_neighborhood(num_nei_cells, search_alpha)

    @property
    def _device(self):
        """Consistent device reference for all operations."""
        return self.primes.device

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key_points = prefix + "map_points"
        key_features = prefix + "map_features"
        key_instance_ids = prefix + "map_instance_ids"

        if key_points in state_dict:
            input_points = state_dict[key_points]
            if input_points.shape != self.map_points.shape:
                self.register_buffer("map_points", torch.empty(input_points.shape, dtype=input_points.dtype, device=self._device))

        if key_instance_ids in state_dict:
            input_instance_ids = state_dict[key_instance_ids]
            if not hasattr(self, "map_instance_ids") or input_instance_ids.shape != self.map_instance_ids.shape:
                self.register_buffer("map_instance_ids", torch.empty(input_instance_ids.shape, dtype=input_instance_ids.dtype, device=self._device))

        if key_features in state_dict:
            input_features = state_dict[key_features]
            if input_features.shape != self.map_features.shape:
                self.map_features = nn.Parameter(torch.empty(input_features.shape, dtype=input_features.dtype, device=self._device))

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _spatial_hash(self, coords):
        # Safely cast to long in case input is float
        if coords.is_floating_point():
            coords = coords.long()
        return ((coords[:, 0] * 73856093) ^ (coords[:, 1] * 19349669) ^ (coords[:, 2] * 83492791)) % self.hash_table_size

    def _init_features(self, num_points, feature_dim, device):
        """
        Initialize features with standard normal distribution.
        """
        feats = torch.zeros(num_points, feature_dim, device=device)
        nn.init.normal_(feats, mean=0.0, std=0.01)
        return feats

    def _get_positional_features(self, coords, feature_dim, scale=1.0):
        """
        Generate Fourier features from coordinates for initialization.
        coords: (N, 3)
        feature_dim: int (must be even)
        """
        assert feature_dim % 2 == 0, "Feature dim must be even for sin/cos split"
        half_dim = feature_dim // 2

        # Generate frequency bands with fixed seed for consistency
        generator = torch.Generator(device=coords.device)
        generator.manual_seed(42)
        freqs = torch.randn(3, half_dim, device=coords.device, generator=generator) * scale
        proj = coords @ freqs  # (N, half_dim)

        # Concatenate sin and cos
        feats = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return feats * 0.1  # Scale down to prevent saturation

    def _initialize_map(self, raw_points, feature_dim):
        """
        Lossless hash registration using Linear Probing.

        Args:
            raw_points: (N, 3) point coordinates
            feature_dim: dimension of learnable features
        """
        if raw_points.shape[0] == 0:
            self.register_buffer("map_points", torch.empty((0, 3), device=self._device))
            self.map_features = nn.Parameter(torch.empty((0, feature_dim), device=self._device))
            self.register_buffer("map_instance_ids", torch.empty((0,), dtype=torch.long, device=self._device))
            # Create hash table
            self.register_buffer("buffer_pt_index", torch.full((self.hash_table_size,), self.EMPTY_KEY, dtype=torch.long))
            self.max_probes = 0
            return

        # (A) Voxelization (Case A: Geometry-only logic)
        grid_coords = torch.floor(raw_points / self.voxel_size).to(torch.int64)
        unique_coords, inverse_indices = torch.unique(grid_coords, dim=0, return_inverse=True)
        num_unique = unique_coords.shape[0]

        if num_unique > self.hash_table_size:
            raise ValueError(f"Unique voxels ({num_unique}) exceed hash table size ({self.hash_table_size})!")

        # (B) Select representative points
        representative_indices = torch.full((num_unique,), -1, dtype=torch.long, device=raw_points.device)
        src_indices = torch.arange(len(raw_points), device=raw_points.device)
        representative_indices.scatter_reduce_(0, inverse_indices, src_indices, reduce="max", include_self=False)

        self.register_buffer("map_points", raw_points[representative_indices])

        # Features initialization using positional features
        init_feats = self._init_features(self.map_points.shape[0], feature_dim, device=self._device)
        # init_feats = self._get_positional_features(self.map_points, feature_dim)
        self.map_features = nn.Parameter(init_feats)
        self.register_buffer("map_instance_ids", torch.full((num_unique,), -1, dtype=torch.long, device=self._device))

        # (C) Create hash table
        self.register_buffer("buffer_pt_index", torch.full((self.hash_table_size,), self.EMPTY_KEY, dtype=torch.long))

        # (D) Insertion
        indices = torch.arange(num_unique, device=unique_coords.device)
        attempts = self._hash_table_insert(unique_coords, indices)

        if attempts > 100:
            print(f"Warning: High collision rate detected (max probes: {attempts}).")

        print(f"[PointMap] Integrated {num_unique} voxels with 0% loss (max probes: {attempts})")
        self.max_probes = attempts

    def _hash_table_insert(self, coords, indices):
        """
        Generic insertion into hash table using Linear Probing.
        Does NOT check for duplicates (assumes filtered input).
        Does NOT store grid coordinates.
        """
        num_items = coords.shape[0]
        if num_items == 0:
            return 0

        # Initial hash calculation
        hash_vals = self._spatial_hash(coords)

        inserted = torch.zeros(num_items, dtype=torch.bool, device=coords.device)
        attempts = 0
        max_attempts = self.hash_table_size

        while not inserted.all():
            if attempts >= max_attempts:
                raise RuntimeError(
                    f"Hash insertion failed after {attempts} probes. Table full or clustering."
                )

            unins = ~inserted

            # Check slots
            curr_h = hash_vals[unins]
            target_pt_idx = self.buffer_pt_index[curr_h]

            # Slot is available if it is EMPTY or DELETED
            available_mask = (target_pt_idx == self.EMPTY_KEY) | (target_pt_idx == self.DELETED_KEY)

            if available_mask.any():
                # Indices into 'unins' subset
                subset_idx = available_mask.nonzero(as_tuple=False).squeeze(1)

                # Indices into original arrays
                # We need to correctly map back.
                # unins is a mask. We want the indices where unins is True.
                unins_indices = unins.nonzero(as_tuple=False).squeeze(1)
                real_indices = unins_indices[subset_idx]

                cand_hash = curr_h[subset_idx]

                # Resolve collisions: if multiple points want same slot, pick first
                sorted_hash, order = torch.sort(cand_hash)
                sorted_items = real_indices[order]

                keep = torch.ones_like(sorted_hash, dtype=torch.bool)
                keep[1:] = sorted_hash[1:] != sorted_hash[:-1]
                chosen_items = sorted_items[keep]
                chosen_slots = hash_vals[chosen_items] # Re-use current hash vals

                # Insert
                self.buffer_pt_index[chosen_slots] = indices[chosen_items]
                inserted[chosen_items] = True

            # Linear probing step for remaining items
            still = ~inserted
            hash_vals[still] = (hash_vals[still] + 1) % self.hash_table_size
            attempts += 1

        return attempts

    def _check_voxel_exists(self, query_coords):
        """
        Check if voxels exist by retrieving points and verifying coordinates.
        Required since buffer_grid_coords is removed.
        """
        N = query_coords.shape[0]
        device = query_coords.device

        hash_vals = self._spatial_hash(query_coords)
        exists = torch.zeros(N, dtype=torch.bool, device=device)
        active_mask = torch.ones(N, dtype=torch.bool, device=device)

        max_probes = getattr(self, 'max_probes', 1000) + self.extra_probe_margin
        curr_hash = hash_vals.clone()

        for _ in range(max_probes):
            if not active_mask.any():
                break

            active_indices = torch.nonzero(active_mask).squeeze(1)
            curr_h = curr_hash[active_indices]

            table_idx = self.buffer_pt_index[curr_h]

            # 1. Empty slot -> Stop
            empty_slots = (table_idx == self.EMPTY_KEY)

            # 2. Check Match (only if not empty and not deleted)
            # Reconstruct coordinate from stored point
            is_valid_entry = (table_idx != self.EMPTY_KEY) & (table_idx != self.DELETED_KEY)

            matches = torch.zeros_like(is_valid_entry)

            # Only compute for valid entries to save ops
            valid_mask_subset = is_valid_entry
            if valid_mask_subset.any():
                valid_idx = table_idx[valid_mask_subset]
                stored_points = self.map_points[valid_idx]
                stored_grid = torch.floor(stored_points / self.voxel_size).to(torch.int64)

                # Compare stored_grid vs query_coords
                # query_coords needs to be indexed by active_indices -> then masked by valid_mask_subset
                query_subset = query_coords[active_indices][valid_mask_subset]

                match_subset = torch.all(stored_grid == query_subset, dim=1)
                matches[valid_mask_subset] = match_subset

            exists[active_indices[matches]] = True

            # Stop if found OR empty
            stop_mask = empty_slots | matches

            active_mask[active_indices[stop_mask]] = False

            # Advance hash
            cont_indices = active_indices[~stop_mask]
            curr_hash[cont_indices] = (curr_hash[cont_indices] + 1) % self.hash_table_size

        return exists

    def register_points(self, new_points, new_features=None, new_instance_ids=None,
                         skip_existing: bool = True):
        """
        Register new points into the map.

        Args:
            new_points: (N, 3) point coordinates
            new_features: Optional pre-computed features. If None, random init.
            new_instance_ids: Optional instance IDs
            skip_existing: If True, deduplicate by voxel and skip existing voxels.
                If False, register all points directly (used for tracking re-registration).
        """
        device = new_points.device
        grid_coords = torch.floor(new_points / self.voxel_size).to(torch.int64)

        if skip_existing:
            # --- Voxel-downsampled registration: one point per new voxel ---
            unique_coords, inverse_indices = torch.unique(grid_coords, dim=0, return_inverse=True)

            exists = self._check_voxel_exists(unique_coords)
            new_mask = ~exists

            if not new_mask.any():
                return 0

            # Get representative point indices for each unique voxel
            representative_indices = torch.full(
                (unique_coords.shape[0],), -1, dtype=torch.long, device=device
            )
            src_indices = torch.arange(len(new_points), device=device)
            representative_indices.scatter_reduce_(
                0, inverse_indices, src_indices, reduce="max", include_self=False
            )

            keep = representative_indices[new_mask]
            new_points = new_points[keep]
            grid_coords = unique_coords[new_mask]
            num_new = new_points.shape[0]

            if new_features is not None:
                new_features = new_features[keep]
            else:
                new_features = self._init_features(num_new, self.map_features.shape[1], device=device)

            if new_instance_ids is not None:
                new_instance_ids = new_instance_ids[keep]
            else:
                new_instance_ids = torch.full((num_new,), -1, dtype=torch.long, device=device)
        else:
            # --- Direct registration: no dedup ---
            num_new = new_points.shape[0]

            if new_features is None:
                new_features = self._init_features(num_new, self.map_features.shape[1], device=device)

            if new_instance_ids is None:
                new_instance_ids = torch.full((num_new,), -1, dtype=torch.long, device=device)

        if (self.map_points.shape[0] + num_new) > self.hash_table_size:
            print(f"Warning: Map size ({self.map_points.shape[0] + num_new}) approaching hash table size")

        # Update Map State
        start_idx = self.map_points.shape[0]
        self.register_buffer("map_points", torch.cat([self.map_points, new_points], dim=0))
        self.map_features = nn.Parameter(torch.cat([self.map_features, new_features], dim=0))
        self.register_buffer("map_instance_ids", torch.cat([self.map_instance_ids, new_instance_ids], dim=0))

        # Insert into Hash Table
        indices_to_insert = torch.arange(start_idx, start_idx + num_new, device=device)
        attempts = self._hash_table_insert(grid_coords, indices_to_insert)
        if attempts > self.max_probes:
            self.max_probes = attempts

        return num_new

    def query_neighbors_from_hash(self, query_points, stop_on_match=True):
        """
        Neighbor query handling multiple points per voxel.
        Collects up to max_candidates candidates per query point to find K nearest.

        Args:
            query_points: (N, 3) query coordinates
            stop_on_match: If True, stop searching after finding a match (for single-point-per-voxel).
                           If False, continue searching to find all points in voxel (for multi-point-per-voxel).
        """
        B = query_points.shape[0]
        device = query_points.device
        query_grid = torch.floor(query_points / self.voxel_size).to(torch.int64)

        neighbor_cells = query_grid.unsqueeze(1) + self.neighbor_dx.unsqueeze(0)
        N_nei = neighbor_cells.shape[1]

        flat_neighbor_cells = neighbor_cells.view(-1, 3)
        hash_vals = self._spatial_hash(flat_neighbor_cells)

        # Buffer to store candidates: (B, max_candidates)
        max_cand = self.max_candidates
        found_candidates = torch.full((B, max_cand), -1, dtype=torch.long, device=device)
        candidate_counts = torch.zeros(B, dtype=torch.long, device=device)

        # Active search mask
        # We have B*N_nei search threads
        active_mask = torch.ones(B * N_nei, dtype=torch.bool, device=device)
        curr_hash = hash_vals.clone()

        # Loop limit
        max_search = getattr(self, 'max_probes', 1000) + self.extra_probe_margin

        for _ in range(max_search):
            if not active_mask.any():
                break

            active_idx = torch.nonzero(active_mask).squeeze(1)
            curr_h = curr_hash[active_idx]

            target_pt_idx = self.buffer_pt_index[curr_h]

            is_empty = (target_pt_idx == self.EMPTY_KEY)
            is_valid = (target_pt_idx != self.EMPTY_KEY) & (target_pt_idx != self.DELETED_KEY)

            # For valid slots, check geometry match
            matches = torch.zeros_like(is_valid, dtype=torch.bool)

            valid_subset_mask = is_valid
            if valid_subset_mask.any():
                valid_pt_indices = target_pt_idx[valid_subset_mask]
                stored_pts = self.map_points[valid_pt_indices]
                stored_grid = torch.floor(stored_pts / self.voxel_size).to(torch.int64)

                # Check match against requested neighbor cell
                # active_idx maps to flat_neighbor_cells
                query_subset = flat_neighbor_cells[active_idx][valid_subset_mask]
                match_subset = torch.all(stored_grid == query_subset, dim=1)
                matches[valid_subset_mask] = match_subset

            # Store matches
            # We need to map active_idx back to batch index B
            # active_idx range [0, B*N_nei)
            if matches.any():
                matched_search_indices = active_idx[matches]
                matched_pt_indices = target_pt_idx[matches]

                batch_indices = matched_search_indices // N_nei

                # Update candidates buffer (vectorized)
                curr_counts = candidate_counts[batch_indices]
                can_add = curr_counts < max_cand

                if can_add.any():
                    valid_b = batch_indices[can_add]
                    valid_c = curr_counts[can_add]
                    valid_p = matched_pt_indices[can_add]

                    found_candidates[valid_b, valid_c] = valid_p
                    candidate_counts[valid_b] += 1

            # Stop condition: Always stop if EMPTY.
            # If stop_on_match=True (single-point-per-voxel), also stop on match.
            # If stop_on_match=False (multi-point-per-voxel), continue to find all points in SAME voxel.
            #   - Skip slots with different voxel's points (hash collision) to avoid full table scan.
            if stop_on_match:
                stop_local_mask = is_empty | matches
            else:
                # Stop on EMPTY or on valid slots that don't match (different voxel due to collision)
                stop_local_mask = is_empty | (is_valid & ~matches)

            indices_to_stop = active_idx[stop_local_mask]
            active_mask[indices_to_stop] = False

            indices_to_continue = active_idx[~stop_local_mask]
            curr_hash[indices_to_continue] = (curr_hash[indices_to_continue] + 1) % self.hash_table_size

        # Now we have candidates. Compute KNN.
        # found_candidates: (B, MAX_CANDIDATES)
        valid_cand_mask = (found_candidates != -1)

        # Gather points - use where instead of clone + mask
        safe_idx = torch.where(valid_cand_mask, found_candidates, torch.zeros_like(found_candidates))

        cand_points = self.map_points[safe_idx]  # (B, MAX_CAND, 3)

        # Dist sq
        dists2 = torch.sum((query_points.unsqueeze(1) - cand_points)**2, dim=-1)

        # Set invalid to infinity
        dists2 = torch.where(valid_cand_mask, dists2, torch.full_like(dists2, float('inf')))

        k = min(self.knn_k, max_cand)
        # Use topk instead of full sort (more efficient for large arrays)
        topk_dist2, topk_idx = torch.topk(dists2, k, dim=1, largest=False)

        topk_dist = torch.sqrt(topk_dist2)
        topk_indices = found_candidates.gather(1, topk_idx)

        return topk_dist, topk_indices

    def set_search_neighborhood(self, num_nei_cells=1, search_alpha=1.0):
        dx_range = torch.arange(-num_nei_cells, num_nei_cells + 1, dtype=torch.int64)
        coords = torch.stack(torch.meshgrid(dx_range, dx_range, dx_range, indexing="ij"), dim=-1)
        dx2 = torch.sum(coords**2, dim=-1)
        self.register_buffer("neighbor_dx", coords[dx2 < (num_nei_cells + search_alpha)**2].view(-1, 3))

    def remove_points_from_hash(self, point_indices):
        """
        Removes points by marking them as DELETED (-2).
        Requires re-calculating hash from map_points.
        """
        if point_indices.numel() == 0:
            return

        points_to_remove = self.map_points[point_indices]
        grid_coords = torch.floor(points_to_remove / self.voxel_size).to(torch.int64)

        curr_hash = self._spatial_hash(grid_coords)
        remaining_mask = torch.ones(point_indices.shape[0], dtype=torch.bool, device=point_indices.device)

        max_search = getattr(self, 'max_probes', 1000) + self.extra_probe_margin

        for _ in range(max_search):
            if not remaining_mask.any():
                break

            active_mask_indices = torch.nonzero(remaining_mask).flatten()
            current_hashes = curr_hash[active_mask_indices]

            table_vals = self.buffer_pt_index[current_hashes]

            target_indices = point_indices[active_mask_indices]

            # Check if we found the exact index
            is_match = (table_vals == target_indices)

            if is_match.any():
                hashes_to_clear = current_hashes[is_match]

                # Mark as DELETED
                self.buffer_pt_index[hashes_to_clear] = self.DELETED_KEY

                # Note: No buffer_grid_coords to clear

                completed_indices = active_mask_indices[is_match]
                remaining_mask[completed_indices] = False

            curr_hash[remaining_mask] = (curr_hash[remaining_mask] + 1) % self.hash_table_size

    def query_instance_ids(self, pts):
        """
        Query instance IDs for points. Uses NN query on the hash table.
        """
        # Find the nearest stored point for each query point
        dist_vals, nn_idx = self.query_neighbors_from_hash(pts)

        # Take the nearest neighbor (k=1)
        nearest_idx = nn_idx[:, 0]
        valid_mask = (dist_vals[:, 0] < 1e10)

        safe_idx = torch.where(valid_mask, nearest_idx, torch.zeros_like(nearest_idx))

        inst_ids = self.map_instance_ids[safe_idx]
        inst_ids = torch.where(valid_mask, inst_ids, torch.full_like(inst_ids, -1))

        return inst_ids

    def interpolate_features(self, pts, temperature=None):
        """
        Interpolate features at query points using distance-weighted KNN.

        Args:
            pts: (N, 3) query points
            temperature: Temperature for softmax attention (defaults to self.query_temperature)

        Returns:
            aggregated_feat: (N, feature_dim) interpolated features
        """
        if temperature is None:
            temperature = self.query_temperature

        dist_vals, nn_idx = self.query_neighbors_from_hash(pts)

        valid_mask = (dist_vals < 1e10)
        safe_nn_idx = torch.where(valid_mask, nn_idx, torch.zeros_like(nn_idx))
        neighbor_feats = self.map_features[safe_nn_idx]

        # Distance-based softmax attention
        masked_dist = torch.where(valid_mask, dist_vals, torch.full_like(dist_vals, 1e10))
        weights = torch.nn.functional.softmax(-masked_dist / temperature, dim=-1)
        aggregated_feat = (neighbor_feats * weights.unsqueeze(-1)).sum(dim=1)

        return aggregated_feat

    def _find_neighbors_chunked(self, points, distance, chunk_size):
        """
        Fallback method: Find neighbor pairs using chunked pairwise distance.
        Used when torch_cluster is not available.
        """
        N = points.shape[0]
        device = points.device
        neighbor_i = []
        neighbor_j = []
        dist_sq_threshold = distance * distance

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_points = points[start:end]  # (chunk, 3)

            # Compute distances from chunk to all points
            diff = chunk_points.unsqueeze(1) - points.unsqueeze(0)  # (chunk, N, 3)
            dist_sq = (diff ** 2).sum(dim=-1)  # (chunk, N)

            # Find pairs within threshold (excluding self)
            mask = (dist_sq < dist_sq_threshold) & (dist_sq > 0)
            chunk_i, chunk_j = torch.where(mask)

            neighbor_i.append(chunk_i + start)
            neighbor_j.append(chunk_j)

        if not neighbor_i:
            return None, None

        return torch.cat(neighbor_i), torch.cat(neighbor_j)

    def refine_instance_ids_by_graph(self, distance=0.02, majority_ratio=0.6, chunk_size=10000):
        """
        Refines instance IDs using a graph based on spatial proximity.
        Uses torch_cluster.radius_graph if available for memory efficiency,
        falls back to chunked pairwise distance otherwise.

        Args:
            distance: Maximum distance to consider points as neighbors
            majority_ratio: Minimum ratio for majority voting to update ID
            chunk_size: Chunk size for pairwise distance computation (fallback memory optimization)
        """
        print(f"[Graph] Building graph for instance ID refinement (dist={distance}, ratio={majority_ratio})...")

        active_mask = (self.buffer_pt_index != self.EMPTY_KEY) & (self.buffer_pt_index != self.DELETED_KEY)
        active_indices = self.buffer_pt_index[active_mask]

        if active_indices.shape[0] == 0:
            print("[Graph] No active points to refine.")
            return

        if not hasattr(self, "map_instance_ids") or self.map_instance_ids.shape[0] == 0:
            print("[Graph] No instance IDs found.")
            return

        device = self.map_points.device
        points = self.map_points[active_indices]  # (N, 3)
        ids = self.map_instance_ids[active_indices]  # (N,)
        N = points.shape[0]

        # Try to use torch_cluster for efficient radius graph construction
        try:
            from torch_cluster import radius_graph
            edge_index = radius_graph(points, r=distance, loop=False)
            neighbor_i = edge_index[0]
            neighbor_j = edge_index[1]
            print("[Graph] Using torch_cluster.radius_graph for neighbor search.")
        except ImportError:
            print("[Graph] torch_cluster not available, using chunked pairwise distance fallback.")
            neighbor_i, neighbor_j = self._find_neighbors_chunked(points, distance, chunk_size)

        if neighbor_i is None or neighbor_i.numel() == 0:
            print("[Graph] No neighbor pairs found within distance threshold.")
            return

        # Build sparse adjacency using scatter operations
        # Count neighbors per point
        neighbor_counts = torch.zeros(N, dtype=torch.long, device=device)
        neighbor_counts.scatter_add_(0, neighbor_i, torch.ones_like(neighbor_i))

        # For majority voting, we need to count instance IDs among neighbors
        # Get neighbor instance IDs
        neighbor_ids = ids[neighbor_j]

        # For each point, find the dominant instance ID among neighbors
        # Remap sparse IDs to compact indices to avoid huge allocation
        unique_ids, inverse = ids.unique(return_inverse=True)
        num_unique = unique_ids.shape[0]
        compact_ids = inverse  # 0 ~ num_unique-1

        id_counts = torch.zeros(N, num_unique, dtype=torch.long, device=device)
        flat_idx = neighbor_i * num_unique + compact_ids[neighbor_j]
        id_counts.view(-1).scatter_add_(0, flat_idx, torch.ones_like(flat_idx))

        # Find dominant ID and its count for each point
        dominant_counts, dominant_compact = id_counts.max(dim=1)
        dominant_ids = unique_ids[dominant_compact]  # compact index -> original ID

        # Calculate ratio
        ratios = dominant_counts.float() / neighbor_counts.float().clamp(min=1)

        # Find points to update
        should_update = (dominant_ids != ids) & (ratios >= majority_ratio) & (neighbor_counts > 0)
        update_mask = should_update

        num_updates = update_mask.sum().item()
        if num_updates > 0:
            new_ids = ids.clone()
            new_ids[update_mask] = dominant_ids[update_mask]
            self.map_instance_ids[active_indices] = new_ids
            print(f"[Graph] Updated {num_updates} instance IDs based on majority voting.")
        else:
            print("[Graph] No instance ID updates required.")
