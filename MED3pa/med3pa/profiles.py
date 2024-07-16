"""
Handles the management and storage of profiles derived from the tree representation. 
It defines a ``Profile`` class to encapsulate metrics and values associated with a specific node in the tree and a ``ProfilesManager`` class to manage collections of profiles and track lost profiles during analysis.
"""
class Profile:
    """
    Represents a profile containing metrics and values associated with a specific node.
    """

    def __init__(self, node_id: int, path: list[str]):
        """
        Initializes a Profile instance with node details and associated metrics.

        Args:
            node_id (int): The identifier for the node associated with this profile.
            path (list[str]): A List OF String containing representation of the path to this node within the tree.
            mean_value (float): The mean confidence score calculated at this profile node.
            metrics (Optional[dict]): Base model metrics associated with this node. Defaults to None.
        """
        self.node_id = node_id
        self.path = path
        self.mean_value = None
        self.metrics = None
        self.node_information = None
        self.detectron_results = None

    def to_dict(self, save_all:bool=True):
        """
        Converts the Profile instance into a dictionary format suitable for serialization.

        Returns:
            dict: A dictionary representation of the Profile instance including the node ID, path, mean value, 
                  metrics, and any detectron results.
        """
        if save_all:
            return {
                'id': self.node_id,
                'path': self.path,
                'metrics': self.metrics,
                'node information': self.node_information, 
                'detectron_results' : self.detectron_results
            }
        else:
            return {
                'id': self.node_id,
                'path': self.path,
            }
    
    def update_detectron_results(self, detectron_results: dict):
        """
        Updates the detectron results associated with this profile.

        Args:
            detectron_results (dict): The results from the Detectron experiment to be added to the profile.
        """
        self.detectron_results = detectron_results

    def update_metrics_results(self, metrics : dict):
        """
        Updates the metrics associated with this profile.

        Args:
            detectron_results (dict): The results from the Detectron experiment to be added to the profile.
        """
        self.metrics = metrics
    
    def update_node_information(self, info : dict):
        """
        Updates the information associated with this profile.

        Args:
            info (dict): The updated node information.
        """
        self.node_information = info



class ProfilesManager:
    """
    Manages the records of profiles and lost profiles based on declaration rates and minimal samples ratio.
    """

    def __init__(self, features:list[str]):
        """
        Initializes the ProfilesManager with a set of features.

        Args:
            features (list[str]): A list of features considered in the profiles.
        """
        self.profiles_records = {}
        self.lost_profiles_records = {}
        self.features = features

    def insert_profiles(self, dr:int, min_samples_ratio:int, profiles: list[Profile]):
        """
        Inserts profiles into the records under a specific dr value and minimum sample ratio.

        Args:
            dr (int): Desired declaration rate as a percentage.
            min_samples_ratio (int): Minimum samples ratio.
            profiles (list[Profile]): The profiles to insert.
        """
        if min_samples_ratio not in self.profiles_records:
            self.profiles_records[min_samples_ratio] = {}
        self.profiles_records[min_samples_ratio][dr] = profiles.copy()

    def insert_lost_profiles(self, dr:int, min_samples_ratio:int, profiles: list[Profile]):
        """
        Inserts lost profiles into the records under a specific dr value and minimum sample ratio.

        Args:
            dr (int): Desired declaration rate as a percentage.
            min_samples_ratio (int): Minimum samples ratio.
            profiles (list[Profile]): The profiles to insert.
        """
        if min_samples_ratio not in self.lost_profiles_records:
            self.lost_profiles_records[min_samples_ratio] = {}
        self.lost_profiles_records[min_samples_ratio][dr] = profiles.copy()    

    def get_profiles(self, min_samples_ratio:int=None, dr:int=None):
        """
        Retrieves profiles based on the specified minimum sample ratio and dr value.
        
        Args:
            dr (int): Desired declaration rate as a percentage.
            min_samples_ratio (int): Minimum samples ratio.
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
        
        Args:
            dr (int): Desired declaration rate as a percentage.
            min_samples_ratio (int): Minimum samples ratio.
        """
        if min_samples_ratio is not None:
            if dr is not None:
                if min_samples_ratio not in self.lost_profiles_records:
                    raise ValueError("The lost profiles for this min_samples_ratio have not been calculated yet!")
                return self.lost_profiles_records[min_samples_ratio][dr]
            return self.lost_profiles_records[min_samples_ratio]
        return self.lost_profiles_records

    @staticmethod
    def transform_to_profiles(profiles_list : list[dict], to_dict:bool=False):
        """
        Transforms a list of profile data into instances of the Profile class or dictionaries.
        
        Args:
            profiles_list (list[dict]): List of profiles data.
            to_dict (bool, optional): If True, transforms profiles to dictionaries. Defaults to True.

        Returns:
            list[Union[dict, Profile]]: List of transformed profiles.
        """
        profiles = []
        for profile in profiles_list:
            if to_dict:
                profile_ins = Profile(profile['node_id'], profile['path']).to_dict()
            else:
                profile_ins = Profile(profile['node_id'], profile['path'])
            profiles.append(profile_ins)
        return profiles
    
    