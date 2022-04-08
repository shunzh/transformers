class GPTKeyValueCache:
    def __init__(self):
        self.cache = {}

    def add(self, input_ids, key_values):
        input_ids_key = tuple(input_ids.tolist())
        if input_ids_key in self.cache.keys():
            return

        keys_to_remove = []
        for cached_key in self.cache.keys():
            if cached_key == input_ids_key[:len(cached_key)]:
                keys_to_remove.append(cached_key)
        for k in keys_to_remove: del self.cache[k]

        self.cache[input_ids_key] = key_values

    def get(self, input_ids):
        """
        Find cached key_values for input_ids.
        Concretely, find the cached sequence which shares the longest prefix with input_ids, and return its key_values.
        Return None if not found.
        """
        if len(self.cache) == 0:
            return None

        input_ids = input_ids.tolist()
        longest_prefix_len = 0
        longest_prefix_key = None

        print(f"searching over dict of size {len(self.cache)}")

        for cached_key in self.cache:
            i = 0
            while i < len(cached_key) and i < len(input_ids) and cached_key[i] == input_ids[i]:
                i += 1

            if i > longest_prefix_len:
                longest_prefix_len = i
                longest_prefix_key = cached_key

        # fixme generate doesn't work if the len of past_key_values is equal to input_ids
        longest_prefix_len = min(longest_prefix_len, len(input_ids) - 1)

        if longest_prefix_len > 0:
            cached_key_values = self.cache[longest_prefix_key]
            key_values = []

            # they only match up to the length of longest_prefix_len, so cut the key_values up to this length
            for layer_past in cached_key_values:
                k, v = layer_past
                key_values.append((k[:, :, :longest_prefix_len, :], v[:, :, :longest_prefix_len, :]))

            print(f"used cache of length {longest_prefix_len}")
            return key_values
        else:
            return None

    def clear(self, encoded_ids):
        encoded_ids = tuple(encoded_ids)
        keys_to_remove = []
        for cached_key in self.cache.keys():
            if cached_key[:len(encoded_ids)] != encoded_ids:
                keys_to_remove.append(cached_key)
        for k in keys_to_remove: del self.cache[k]
