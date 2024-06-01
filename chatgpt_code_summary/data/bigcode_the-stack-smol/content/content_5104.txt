from collections import defaultdict
from typing import Dict, Tuple, Iterator, Callable, Any, Optional
from dataclasses import dataclass

"""
Provides the `TaggedProfiler` class related to record profiling.
TODO: Better description needed.
"""

@dataclass
class TaggedProfilerRecordStatus:
    offset: int
    tag: str
    key: str
    val: Any 
    r: Optional[dict]

@dataclass
class TaggedProfilerSummary:
    total: int
    histo: dict 
    index: Optional[dict]
    cache: Optional[dict]

    def describe(self) -> Iterator[str]:
        yield f"histo = {self.histo}"
        if self.index is None:
            yield f"index = {self.index}"
        else:
            yield f"index with {len(self.index)} items:"
            for label, nums in self.index.items():
                yield f"label = '{label}, size = {len(nums)}:"
                if self.cache is not None:
                    for n in nums:
                        yield f"cache[{n}] = {self.cache[n]}"


class TaggedProfiler:
    """A useful tag-based profiler class which we'll describe when we have more time."""

    def __init__(self, tagmap: Dict[str,Callable]):
        self.tagmap = tagmap

    def eval_dict(self, r: dict) -> Iterator[Tuple[str,str,str]]:
        for (tag, f) in self.tagmap.items(): 
            for (k, v) in r.items():
                if f(v):
                    yield (tag, k, v)

    def evaluate(self, recs: Iterator[dict], deep: bool = False) -> Iterator[TaggedProfilerRecordStatus]:
        for (i, r) in enumerate(recs):
            for (tag, k, v) in self.eval_dict(r):
                yield TaggedProfilerRecordStatus(i, tag, k, v, r if deep else None)

    def profile(self, recs: Iterator[dict], index: bool = False, deep: bool = False) -> TaggedProfilerSummary:
        """Provides the most useful summary counts you'll likely want from the incoming record sequence.
        Optional :index and :deep flags allow us to return special indexing and cachinc structs which we'll describe later."""
        # We use underscores for all "recording" structures.
        # Non-nunderscore names for input variables and flags.
        labels = list(self.tagmap.keys())
        temp_cache: Dict[int,Any] = {}
        temp_index: Dict[str,Any] = {k:defaultdict(int) for k in labels}
        for status in self.evaluate(recs, deep): 
            temp_cache[status.offset] = status.r if deep else 1
            temp_index[status.tag][status.offset] += 1
        _total = len(temp_cache)
        _histo: Dict[str,int] = {k:len(v) for (k,v) in temp_index.items()}
        _index: Optional[Dict[str,list]] = None
        _cache: Optional[Dict[int,Any]] = None
        if temp_index:
            _index = {k:list(v.keys()) for k,v in temp_index.items()}
        if deep: 
            _cache = temp_cache
        return TaggedProfilerSummary(_total, _histo, _index, _cache)

