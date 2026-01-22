
#Should've done this in torchRL
class FeatureMapping:
    def __init__(self, feature_map, internal_state_ini, update_internal_state):
        self.feature_map = feature_map
        self.internal_state_ini = internal_state_ini
        self.update_internal_state = update_internal_state

    def get_features(self, state, internal_state):
        self.update_internal_state(state, internal_state)
        return self.feature_map(state, internal_state)

    def initialize_internal_state(self, state):
        return self.internal_state_ini(state)

#if absolutely necessary we can define a set of features - right now this hasn't been implemented in the batch-handling/optimization, so would bug out.
class MultiFeature:
    def __init__(self, feature_tuple):
        self.feature_tuple = feature_tuple
        self.feature_number = len(feature_tuple)

    def get_features(self, state, feature_states):
        for i in range(self.feature_number):
            feature_states[i] = self.feature_tuple[i].update_feature_state(state, feature_states[i])        
        return tuple(self.feature_tuple[i].get_features(state, feature_states[i]) for i in range(self.feature_number))

    def initialize_internal_state(self, state):
        return [self.feature_tuple[i].internal_state(state) for i in range(self.feature_number)]
