import torch


class GPTKeyValueCache:
    def __init__(self):
        """
        A cache dict, self.cache[input_ids] = key values
        """
        self.cache = {}

    def add(self, input_ids_list, key_values):
        input_ids_list = input_ids_list.tolist()

        for i, input_ids in enumerate(input_ids_list):
            input_ids = tuple(input_ids)

            if input_ids in self.cache.keys():
                continue
            else:
                new_key_values = []

                for layer in key_values:
                    k, v = layer
                    new_key_values.append((k[i, :, :, :], v[i, :, :, :]))

                self.cache[input_ids] = new_key_values

    def get(self, input_ids_list):
        """
        Find cached key_values for input_ids.
        Concretely, find the cached sequence which shares the longest prefix with input_ids, and return its key_values.
        Return None if not found.
        """
        if len(self.cache) == 0:
            return None

        print(f"searching over dict of size {len(self.cache)}")

        input_ids_list = input_ids_list.tolist()

        longest_prefix_lens = []
        longest_prefix_keys = []
        for i, input_ids in enumerate(input_ids_list):
            longest_prefix_len = 0
            for cached_key in self.cache.keys():
                i = 0
                while i < len(cached_key) and i < len(input_ids) and cached_key[i] == input_ids[i]:
                    i += 1

                if i > longest_prefix_len:
                    longest_prefix_len = i
                    longest_prefix_key = cached_key

            # fixme generate doesn't work if the len of past_key_values is equal to input_ids
            longest_prefix_lens.append(min(longest_prefix_len, len(input_ids) - 1))
            longest_prefix_keys.append(longest_prefix_key)

        longest_prefix_len_for_all = min(longest_prefix_lens)

        if longest_prefix_len_for_all > 0:
            cached_key_values = [self.cache[prefix_key] for prefix_key in longest_prefix_keys]
            ret_key_values = []

            for layer_idx in range(len(cached_key_values[0])):
                layer_keys = []
                layer_values = []
                for key_values in cached_key_values:
                    k, v = key_values[layer_idx]
                    layer_keys.append(k[:, :longest_prefix_len_for_all, :])
                    layer_values.append(v[:, :longest_prefix_len_for_all, :])
                ret_key_values.append((torch.stack(layer_keys, dim=0), torch.stack(layer_values, dim=0)))

            print(f"used cache of length {longest_prefix_len_for_all}")
            return ret_key_values
        else:
            return None

    def clear(self, encoded_ids=None):
        if encoded_ids is None:
            # clear cache unconditionally
            self.cache = {}
        else:
            encoded_ids = tuple(encoded_ids)
            keys_to_remove = []
            for cached_key in self.cache.keys():
                if cached_key[:len(encoded_ids)] != encoded_ids:
                    keys_to_remove.append(cached_key)
            for k in keys_to_remove: del self.cache[k]


class GPTTopOutputCache:
    def __init__(self, k):
        """
        A cache dict, self.cache[input_ids] = [the i-th most likely token output, for i in range(k)]
        """
        self.k = k
        self.cache = {}

    def add(self, input_ids_list, scores):
        input_ids_list = input_ids_list.tolist()

        for i, input_ids in enumerate(input_ids_list):
            input_ids = tuple(input_ids)

            if input_ids not in self.cache.keys():
                top_k_scores, top_k_tokens = torch.topk(scores[i, :], k=self.k, sorted=True)
                self.cache[input_ids] = top_k_tokens.tolist()

    def get(self, input_ids):
        input_ids = tuple(input_ids)

        if input_ids in self.cache.keys():
            return self.cache[input_ids]
        else:
            return None

    def clear(self, encoded_ids=None):
        if encoded_ids is None:
            # clear cache unconditionally
            self.cache = {}
        else:
            encoded_ids = tuple(encoded_ids)
            keys_to_remove = []
            for cached_key in self.cache.keys():
                if cached_key[:len(encoded_ids)] != encoded_ids:
                    keys_to_remove.append(cached_key)
            for k in keys_to_remove: del self.cache[k]
