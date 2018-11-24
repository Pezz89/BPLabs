from pathops import dir_must_exist
import os
import dill

class Participant:
    def __init__(self, participant_dir=None, number=None, age=None, gender=None, handedness=None, general_notes=None):
        '''
        '''
        dir_must_exist(participant_dir)
        self.participant_dir = participant_dir
        self.data_paths = {}
        self.generate_folder_hierachy()

        self.data = {
            "info": {
                "number": number,
                "age": age,
                "gender": gender,
                "handedness": handedness,
                "general_notes": general_notes
            },

            "adaptive_matrix_data": {
                "notes": ''
            },

            "set_matrix_data": {
                "notes": ''
            },

            "da_data": {
                "notes": ''
            },

            "click_data": {
                "notes": ''
            },

            "pta_data": {
                "notes": ''
            }
        }

    def generate_folder_hierachy(self):
        '''
        '''
        sub_dirs = ["adaptive_matrix_data", "da_data", "pta_data", "click_data", "info", "set_matrix_data"]
        for dir_name in sub_dirs:
            path = os.path.join(self.participant_dir, dir_name)
            dir_must_exist(path)
            self.data_paths[dir_name] = path

    def __setitem__(self, key, item):
        self.data[key] = item

    def __getitem__(self, key):
        return self.data[key]

    def set_info(self, info):
        self.data['info'] = info

    def save(self, data_key):
        '''
        '''
        directory = self.data_paths[data_key]
        with open(os.path.join(directory, '{}.pkl'.format(data_key)), 'wb') as f:
            dill.dump(self.data[data_key], f)


    def load(self, data_key):
        '''
        '''
        folder = os.path.join(self.participant_dir, data_key)
        with open(os.path.join(folder, "{}.pkl".format(data_key)), 'rb') as f:
            self.data[data_key].update(dill.load(f))


