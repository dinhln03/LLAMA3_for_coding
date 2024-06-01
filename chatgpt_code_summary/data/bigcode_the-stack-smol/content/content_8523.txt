from typing import Sequence, Dict, List, Optional
from abc import ABC, abstractmethod

# from nltk.corpus import wordnet
from pymagnitude import Magnitude

from .utils import (
    UPPERCASE_RE,
    LOWERCASE_RE,
    DIGIT_RE,
    PUNC_REPEAT_RE,
)

class FeatureExtractor(ABC):
    @abstractmethod
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        raise NotImplementedError

class BiasFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        features["bias"] = 1.0

class TokenFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        features["tok[%d]=%s" % (relative_idx, token)] = 1.0

class UppercaseFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if str.isupper(token):
            features["uppercase[%d]" % (relative_idx)] = 1.0

class TitlecaseFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if str.istitle(token):
            features["titlecase[%d]" % (relative_idx)] = 1.0

class InitialTitlecaseFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if str.istitle(token) and current_idx + relative_idx == 0:
            features["initialtitlecase[%d]" % (relative_idx)] = 1.0

class PunctuationFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if PUNC_REPEAT_RE.fullmatch(token):
            features["punc[%d]" % (relative_idx)] = 1.0

class DigitFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if DIGIT_RE.search(token):
            features["digit[%d]" % (relative_idx)] = 1.0

class WordShapeFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        shape_str = token
        shape_str = UPPERCASE_RE.sub("X", shape_str)
        shape_str = LOWERCASE_RE.sub("x", shape_str)
        shape_str = DIGIT_RE.sub("0", shape_str)
        features["shape[%d]=%s" % (relative_idx, shape_str)] = 1.0

"""
class LikelyAdjectiveFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        for synset in wordnet.synsets(token):
            if synset.name().split('.')[0] == token:
                if synset.pos() == 's':
                    features["adj[%d]" % (relative_idx)] = 1.0

class AfterVerbFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if relative_idx != -1:
            return

        for synset in wordnet.synsets(token):
            if synset.name().split('.')[0] == token:
                if synset.pos() == 'v':
                    features["afterverb[%d]" % (relative_idx)] = 1.0

class PosFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        for synset in wordnet.synsets(token):
            if synset.name().split('.')[0] == token:
                features[f"adj[{relative_idx}]={synset.pos()}"] = 1.0
"""

class WordVectorFeature(FeatureExtractor):
    def __init__(self, vectors_path: str, scaling: float = 1.0) -> None:
        self.vectors = Magnitude(vectors_path, normalized=False)
        self.scaling = scaling

    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        vector = self.vectors.query(token)
        keys = ["v" + str(i) for i in range(len(vector))]
        values = vector * self.scaling
        features.update(zip(keys, values))


class BrownClusterFeature(FeatureExtractor):
    def __init__(
        self,
        clusters_path: str,
        *,
        use_full_paths: bool = False,
        use_prefixes: bool = False,
        prefixes: Optional[Sequence[int]] = None,
    ):
        if not use_full_paths and not use_prefixes:
            raise ValueError
        self.prefix_dict = {}
        self.full_path_dict = {}
        self.use_full_paths = use_full_paths
        with open(clusters_path, "r", encoding='utf-8') as cluster_file:
            for line in cluster_file:
                cluster, word, _frequency = line.split("\t")
                if use_full_paths:
                    self.full_path_dict[word] = cluster
                elif use_prefixes:
                    if prefixes is not None:
                        self.prefix_dict[word] = [
                            cluster[:prefix]
                            for prefix in prefixes
                            if prefix <= len(cluster)
                        ]
                    else:
                        self.prefix_dict[word] = [
                            cluster[:prefix] for prefix in range(1, len(cluster) + 1)
                        ]

    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if relative_idx != 0:
            return
        if self.use_full_paths and token in self.full_path_dict:
            features[f"cpath={self.full_path_dict[token]}"] = 1.0
        elif token in self.prefix_dict:
            for prefix in self.prefix_dict[token]:
                features[f"cprefix{len(prefix)}={prefix}"] = 1.0

class WindowedTokenFeatureExtractor:
    def __init__(self, feature_extractors: Sequence[FeatureExtractor], window_size: int):
        self.feature_extractors = feature_extractors
        self.window_size = window_size

    def extract(self, tokens: Sequence[str]) -> List[Dict[str, float]]:
        features = []
        for i, _ in enumerate(tokens):
            current_feature = {}
            start = max(0, i - self.window_size)
            end = min(len(tokens) - 1, i + self.window_size) + 1
            for j in range(start, end):
                relative_idx = j - i
                for feature_extractor in self.feature_extractors:
                    feature_extractor.extract(
                        tokens[j], i, relative_idx, tokens, current_feature
                    )
            features.append(current_feature)
        return features

