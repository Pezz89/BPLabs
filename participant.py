from pathops import dir_must_exist
import os

class Participant:
    def __init__(self, participant_dir=None, number=None, age=None, gender=None, general_notes=None):
        '''
        '''
        dir_must_exist(participant_dir)
        self.participant_dir = participant_dir
        self.generate_folder_hierachy()

        self.info = {
            "number": number,
            "age": age,
            "gender": gender,
            "general_notes": general_notes
        }

        self.adaptive_matrix_data = {
            "notes": ''
        }

        self.set_matrix_data = {
            "notes": ''
        }

        self.da_data = {
            "notes": ''
        }

        self.click_data = {
            "notes": ''
        }

        self.pta_data = {
            "notes": ''
        }

    def generate_folder_hierachy(self):
        '''
        '''
        sub_dirs = ["adaptive_matrix_data", "da_data", "pta_data", "click_data", "info", "set_matrix_data"]
        for dir_name in sub_dirs:
            dir_must_exist(os.path.join(self.participant_dir, dir_name))


    def save(self, folder):
        '''
        '''

    def load(self, folder):
        '''
        '''
