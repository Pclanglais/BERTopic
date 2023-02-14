import openai
import numpy as np
import pandas as pd
import time
from scipy.sparse import csr_matrix
from typing import Mapping, List, Tuple, Union, Any
from sklearn.metrics.pairwise import cosine_similarity
from bertopic.representation._base import BaseRepresentation


DEFAULT_PROMPT = """
Voici une liste de textes où chaque collection de textes décrit un topic. Après chaque collection de textes, le nom du topic qu'ils représentent est mentionné en tant que titre court et hautement descriptif.

Topic:
Sample texts from this topic:
- Les régimes traditionnels dans la plupart des cultures étaient principalement à base de plantes avec un peu de viande en plus, mais avec l'essor de la production de viande de style industriel et de l'élevage en usine, la viande est devenue un aliment de base.
- La viande, mais surtout le boeuf, est le mot aliment en termes d'émissions.
- Manger de la viande ne vous rend pas une mauvaise personne, ne pas manger de viande ne vous rend pas une bonne personne.

Keywords: viande boeuf manger manger émissions steak nourriture santé transformé poulet
Topic name: Impacts environnementaux de la consommation de viande
---
Topic:
Sample texts from this topic:
- J'ai commandé le produit il y a des semaines, mais il n'est toujours pas arrivé!
- Le site mentionne qu'il ne faut que quelques jours pour livrer, mais je n'ai toujours rien reçu.
- J'ai reçu un message indiquant que j'ai reçu le moniteur, mais ce n'est pas vrai!
- Il a fallu un mois de plus pour livrer que ce qui était conseillé...

Keywords: livreur semaines produit expédition longue livraison reçu arrivé arriver semaine
Topic name: Problèmes d'expédition et de livraison
---
"""


class OpenAI(BaseRepresentation):
    """ Using the OpenAI API to generate topic labels based
    on one of their GPT-3 models. For an overview see:

    https://platform.openai.com/docs/models/gpt-3

    Arguments:
        generator_kwargs: Kwargs passed to `openai.Completion.create`
                          for fine-tuning the output.
        prompt: The prompt to be used in the model. If no prompt is given,
                `self.default_prompt_` is used instead.
                NOTE: Use `"[KEYWORDS]"` and `"[DOCUMENTS]"` in the prompt
                to decide where the keywords and documents need to be
                inserted.

    Usage:

    To use this, you will need to install the openai package first:

    `pip install openai`

    Then, get yourself an API key and use OpenAI's API as follows:

    ```python
    import openai
    from bertopic.representation import OpenAI
    from bertopic import BERTopic

    # Create your representation model
    representation_model = OpenAI()

    # Use the representation model in BERTopic on top of the default pipeline
    topic_model = BERTopic(representation_model=representation_model)
    ```

    You can also use a custom prompt:

    ```python
    prompt = "I have the following documents: [DOCUMENTS]. What topic do they contain?"
    representation_model = OpenAI(prompt=prompt)
    ```
    """
    def __init__(self,
                 model: str = "text-ada-001",
                 prompt: str = None,
                 generator_kwargs: Mapping[str, Any] = {},
                 ):
        self.model = model
        self.prompt = prompt if prompt is not None else DEFAULT_PROMPT
        self.default_prompt_ = DEFAULT_PROMPT

        self.generator_kwargs = generator_kwargs
        if self.generator_kwargs.get("model"):
            self.model = generator_kwargs.get("model")
        if self.generator_kwargs.get("prompt"):
            del self.generator_kwargs["prompt"]
        if not self.generator_kwargs.get("stop"):
            self.generator_kwargs["stop"] = "\n"

    def extract_topics(self,
                       topic_model,
                       documents: pd.DataFrame,
                       c_tf_idf: csr_matrix,
                       topics: Mapping[str, List[Tuple[str, float]]]
                       ) -> Mapping[str, List[Tuple[str, float]]]:
        """ Extract topics

        Arguments:
            topic_model: A BERTopic model
            documents: All input documents
            c_tf_idf: The topic c-TF-IDF representation
            topics: The candidate topics as calculated with c-TF-IDF

        Returns:
            updated_topics: Updated topic representations
        """
        # Extract the top 4 representative documents per topic
        repr_docs_mappings, _, _ = topic_model._extract_representative_docs(c_tf_idf, documents, topics, 500, 4)

        # Generate using OpenAI's Language Model
        updated_topics = {}
        for topic, docs in repr_docs_mappings.items():
            prompt = self._create_prompt(docs, topic, topics)
            response = openai.Completion.create(model=self.model, prompt=prompt, **self.generator_kwargs)
            label = response["choices"][0]["text"].strip()
            print(label)
            updated_topics[topic] = [(label, 1)] + [("", 0) for _ in range(9)]
            time.sleep(10)

        return updated_topics

    def _create_prompt(self, docs, topic, topics):
        keywords = list(zip(*topics[topic]))[0]

        # Use a prompt that leverages either keywords or documents in
        # a custom location
        prompt = ""
        if "[KEYWORDS]" in self.prompt:
            prompt += self.prompt.replace("[KEYWORDS]", " ".join(keywords))
        if "[DOCUMENTS]" in self.prompt:
            to_replace = ""
            for doc in docs:
                to_replace += f"- {doc[:255]}\n"
            prompt += self.prompt.replace("[DOCUMENTS]", to_replace)

        # Use the default prompt
        if "[KEYWORDS]" and "[DOCUMENTS]" not in self.prompt:
            prompt = self.prompt + 'Topic:\nSample texts from this topic:\n'
            for doc in docs:
                prompt += f"- {doc[:255]}\n"
            prompt += "Keywords: " + " ".join(keywords)
            prompt += "\nTopic name:"
        return prompt
