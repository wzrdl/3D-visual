import torch
from typing import List # in order to make sure that the keywords input is a list

from sentence_transformers import util
from app.client_data_manager import ClientDataManager

class SceneBrain():
    def __init__(self):
        self.keywords = []
        self.top_matches = []
        self.stopwords = set()
        self.stop_words_set()

        # as these are used often
        self.cdm = ClientDataManager() # cdm for ease
        self.embedder = self.cdm.miniM_model

    def stop_words_set(self):

        file = "stopwords.txt"
        myFile = open(file, "r")
        for line in myFile:
            self.stopwords.add(line.strip())
        myFile.close()

        return

    # takes user input and finds keywords from it
    def extract_keywords(self, user_input):
        """
        From user input, separates keywords that we can later use
        """

        user_input = user_input.lower()
        user_input = user_input.strip()
        split_input = user_input.split(" ") # splits based on a space

        # just in case there are irregular spaces
        for word in split_input:
            split_input[word] = split_input[word].strip()

        clean_list = []

        """ 
        OPTIONAL
        sentence transformer SHOULD be able to tell articles apart
        If we wanted a list that is usable in other contexts, though, this could be userufl
        """
        remove_stopwords = True  # adjust this for what we want

        if remove_stopwords == True:
            for word in split_input:
                if word not in self.stop_words_set():
                    clean_list.append(word)
        else:
            for word in split_input:
                clean_list.append(word)

        return clean_list

    def keywords_to_vector(self, user_input: List[str]):
        self. keywords = self.extract_keywords(user_input)

        queries = self.keywords
        query_embedding = self.embedder.encode_query(queries)

        # seeing the scores or calculations for the test
        similarity_score = util.cos_sim(query_embedding, self.cdm.vector_database)
        # makes it a 2D grid with the first [0] being the query and the second holding the
        #       different temp_data lines

        """ likely need edits here especially """
        scores, indices = torch.topk(similarity_score[0])

        # checking which scores are more than 0.5
        for score, index in zip(scores, indices):
            if score > 0.5:
                self.top_matches.append(self.keywords[index])
                print(f"MATCH: {self.keywords[index]} has a score of ({score:4f})")

        # if there is no good match -- prevents error
        if len(self.top_matches) == 0:
            print("No matches found, placeholder set as match.")
            self.top_matches.append("placeholder")

        return self.top_matches