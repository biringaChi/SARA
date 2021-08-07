import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

from typing import Dict, List, Tuple, Union, Set, TypeVar
from utils.util import FeatureVec, HandleCodeRepo, Sourcecode
import numpy as np
import csv
import re
from math import log10
import javalang
from gensim import models
from javalang.ast import Node
from javalang.tree import CompilationUnit
from javalang.parser import JavaParserError, JavaSyntaxError


class Helper(HandleCodeRepo):
    def __init__(self) -> None:
        super().__init__()

    def reg_ex_pattern(self) -> Union[Dict[str, str], Dict[str, Set[str]]]:
        return {
            "SPACES": " +",
            "TABS": "\\t+",
            "WORDS": "\w+",
            "COMMENTS": "/\*(.|[\r\n])*?\*/|//.*",
            "KEYWORDS": {
                    "abstract", "assert", "boolean", "break", "byte", "case", "catch",
                    "char", "class", "const*", "**", "***", "****", "continue", "default", "do", "double",
                    "else", "enum", "extends", "final", "finally", "float", "for", "goto*", "if",
					"implements", "import", "instanceof", "int", "interface", "long", "native",
                    "new", "package", "private", "protected", "public", "return", "short", "static",
                    "strictfp**", "super", "switch", "synchronized", "this", "throw", "throws",
                    "transient", "try", "void", "volatile", "while"
            }
        }

    def get_file_length(self) -> List[int]:
        return [self.__len__(file) for file in self.get_sourcecode()]

    def modify_sourcecode(self) -> Sourcecode:
        modified = []
        for code in self.get_sourcecode():
            temp = re.sub(self.reg_ex_pattern().get("COMMENTS"), "", code)
            modified.append(temp)
        return modified

    def feature_extractor(self, *args) -> List[float]:
        features = []
        if self.__len__(args) >= 1:
            for feature_arg in args:
                temp = []
                for file_length, feature_frequency in zip(self.get_file_length(), feature_arg):
                    with np.errstate(divide="ignore"):
                        temp.append(-np.log10(feature_frequency / file_length))
                features.append(temp)
            return features
        raise ValueError(
            "Invalid argument. Only a feature sequence is accepted")


class Layout(Helper):
    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return self.__str__()

    def get_frequency(self) -> Tuple[List[float]]:
        emptylines, codelines, spaces, tabs = ([] for _ in range(4))

        for sourcecode in self.get_sourcecode():
            emptylines.append(self.__len__([emptyline for emptyline in sourcecode.splitlines() if emptyline == ""]))
            codelines.append(self.__len__([self.__len__(codeline) for codeline in sourcecode.splitlines() if codeline.strip()]))
            spaces.append(self.__len__(re.findall(self.reg_ex_pattern().get("SPACES"), sourcecode)))
            tabs.append(self.__len__(re.findall(self.reg_ex_pattern().get("TABS"), sourcecode)))

        return emptylines, codelines, spaces, tabs


class Lexical(Helper):
    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return self.__str__()

    def get_frequency(self) -> Tuple[List[float]]:
        trees, _ = self.get_trees(self.get_sourcecode())
        imports, comments, keywords, methods = ([] for _ in range(4))

        for tree, sourcecode, modified_code in zip(trees, self.get_sourcecode(), self.modify_sourcecode()):
            imports.append(self.__len__([True for _, node in tree if node.__class__.__name__ in "Import"]))
            comments.append(self.__len__(re.findall(self.reg_ex_pattern().get("COMMENTS"), sourcecode)))
            keywords.append(self.__len__([True for code in modified_code.split() if code in self.reg_ex_pattern().get("KEYWORDS")]))
            methods.append(self.__len__([True for _, node in tree if node.__class__.__name__ in "MethodDeclaration" or "MethodReference"]))

        return imports, comments, keywords, methods


class Syntactic(Helper):
    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return self.__str__()


class Extractor(Helper):
    def __init__(self) -> None:
        super().__init__()

    def extract(self):
        emptylines, codelines, spaces, tabs = Layout().get_frequency()
        imports, comments, keywords, methods = Lexical().get_frequency()
        emptylines, codelines, spaces, tabs, imports, comments, keywords, methods = self.feature_extractor(
        emptylines, codelines, spaces, tabs, imports, comments, keywords, methods)
        
        
        return {
            "emptylines" : emptylines,
            "codelines" : codelines,
            "spaces" : spaces,
            "tabs" : tabs,
            "imports" : imports,
            "comments" : comments,
            "keywords" : keywords,
            "methods" : methods
    	}
        