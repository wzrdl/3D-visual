import torch
from typing import List # in order to make sure that the keywords input is a list

from sentence_transformers import util
from app.client_data_manager import ClientDataManager

class SceneBrain():
    def __init__(self):
        self.keywords = []
        self.top_matches = []
        self.stopwords = set()

        # as these are used often
        self.cdm = ClientDataManager() # cdm for ease
        self.embedder = self.cdm.miniM_model

    def stop_words_set(self):

        file = "assets/stopwords.txt"
        myFile = open(file, "r")
        for line in myFile:
            self.stopwords.add(line.strip())
        myFile.close()

        return self.stopwords

    # takes user input and finds keywords from it
    def extract_keywords(self, user_input: str):
        """
        From user input, separates keywords that we can later use
        """

        user_input = user_input.lower()
        user_input = user_input.strip()
        split_input = user_input.split(" ") # splits based on a space

        # just in case there are irregular spaces
        for i in range(len(split_input)-1):
            split_input[i] = split_input[i].strip()

        clean_list = []

        """ 
        OPTIONAL
        sentence transformer SHOULD be able to tell articles apart
        If we wanted a list that is usable in other contexts, though, this could be userufl
        """
        remove_stopwords = True  # adjust this for what we want

        if remove_stopwords == True:
            self.stopwords = self.stop_words_set()

            for i in range(len(split_input) - 1):
                if split_input[i] not in self.stop_words_set():
                    clean_list.append(split_input[i])
        else:
            for i in range(len(split_input) - 1):
                clean_list.append(split_input[i])

        return clean_list

    def keywords_to_vector(self, user_input: str):
        self. keywords = self.extract_keywords(user_input)

        queries = " ".join(self.keywords)
        query_embedding = self.embedder.encode_query(queries)

        # seeing the scores or calculations for the test
        similarity_score = util.cos_sim(query_embedding, self.cdm.vector_database)
        # makes it a 2D grid with the first [0] being the query and the second holding the
        #       different temp_data lines

        all_names = self.cdm.name_order
        k = len(all_names)

        if k > 0:
            scores, indices = torch.topk(similarity_score[0], k=k)
        else:
            return ["placeholder"]

        """Edit this as we see fit"""
        match_accuracy = 0.4
        # 0.5 was too high for the pytest

        # checking which scores are more than 0.5
        for score, index in zip(scores, indices):
            if score > match_accuracy:
                match_name = self.cdm.name_order[index]
                self.top_matches.append(match_name)
                print(f"MATCH: {match_name} has a score of ({score:4f})")

        # if there is no good match -- prevents error
        if len(self.top_matches) == 0:
            print("No matches found, placeholder set as match.")
            self.top_matches.append("placeholder")

        return self.top_matches