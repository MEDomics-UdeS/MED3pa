"""
this module handles and manages the profiles information.
"""
class Profile:
    """
    Represents a profile containing metrics and values associated with a specific node.
    """

    def __init__(self, node_id, path, mean_value, metrics=None):
        """
        Initializes a Profile instance with node details and associated metrics.

        Args:
            node_id (int): The identifier for the node associated with this profile.
            path (str): A string representation of the path to this node within the tree.
            mean_value (float): The mean value calculated at this node.
            metrics (Optional[dict]): Base model metrics associated with this node. Defaults to None.
        """
        self.node_id = node_id
        self.path = path
        self.mean_value = mean_value
        self.metrics = metrics
        self.detectron_results = None

    def to_dict(self):
        """
        Converts the Profile instance into a dictionary format suitable for serialization.

        Returns:
            dict: A dictionary representation of the Profile instance including the node ID, path, mean value, 
                  metrics, and any detectron results.
        """
        return {
            'id': self.node_id,
            'path': self.path,
            'value': self.mean_value,
            'metrics': self.metrics, 
            'detectron_results' : self.detectron_results
        }
    
    def update_detectron_results(self, detectron_results):
        """
        Updates the detectron results associated with this profile.

        Args:
            detectron_results (dict): The results from the Detectron experiment to be added to the profile.
        """
        self.detectron_results = detectron_results

class ProfilesManager:
    """
    Manages the records of profiles and lost profiles based on declaration rates and minimal samples ratio.
    """

    def __init__(self, features):
        """
        Initializes the ProfilesManager with a set of features.

        Args:
            features (list): A list of features considered in the profiles.
        """
        self.profiles_records = {}
        self.lost_profiles_records = {}
        self.features = features

    def insert_profiles(self, dr, min_samples_ratio, profiles):
        """
        Inserts profiles into the records under a specific dr value and minimum sample ratio.
        """
        if min_samples_ratio not in self.profiles_records:
            self.profiles_records[min_samples_ratio] = {}
        self.profiles_records[min_samples_ratio][dr] = profiles

    def insert_lost_profiles(self, dr, min_samples_ratio, profiles):
        """
        Inserts lost profiles into the records under a specific dr value and minimum sample ratio.
        """
        if min_samples_ratio not in self.lost_profiles_records:
            self.lost_profiles_records[min_samples_ratio] = {}
        self.lost_profiles_records[min_samples_ratio][dr] = profiles

    def get_profiles(self, min_samples_ratio=None, dr=None):
        """
        Retrieves profiles based on the specified minimum sample ratio and dr value.
        """
        if min_samples_ratio is not None:
            if dr is not None:
                if min_samples_ratio not in self.profiles_records:
                    raise ValueError("The profiles for this min_samples_ratio have not been calculated yet!")
                return self.profiles_records[min_samples_ratio][dr]
            return self.profiles_records[min_samples_ratio]
        return self.profiles_records

    def get_lost_profiles(self, min_samples_ratio=None, dr=None):
        """
        Retrieves lost profiles based on the specified minimum sample ratio and dr value.
        """
        if min_samples_ratio is not None:
            if dr is not None:
                if min_samples_ratio not in self.lost_profiles_records:
                    raise ValueError("The lost profiles for this min_samples_ratio have not been calculated yet!")
                return self.lost_profiles_records[min_samples_ratio][dr]
            return self.lost_profiles_records[min_samples_ratio]
        return self.lost_profiles_records

    @staticmethod
    def transform_to_profiles(profiles_list, to_dict=True):
        """
        Transforms a list of profile data into instances of the Profile class or dictionaries.
        """
        profiles = []
        for profile in profiles_list:
            if to_dict:
                profile_ins = Profile(profile['id'], profile['path'], profile['value']).to_dict()
            else:
                profile_ins = Profile(profile['id'], profile['path'], profile['value'])
            profiles.append(profile_ins)
        return profiles