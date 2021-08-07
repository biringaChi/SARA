import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
from gensim import models
from javalang.ast import Node
from javalang.tree import CompilationUnit
from javalang.parser import JavaParserError, JavaSyntaxError

import csv
import numpy as np
import javalang
from typing import Dict, List, Sequence, Text, Tuple, Union, Set
from nltk import RegexpTokenizer
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from utils.util import HandleCodeRepo
import matplotlib.pyplot as plt

"""
Creates an Abstract Syntax Tree (AST) embedding structure.
"""


class TreeEmbeddings(HandleCodeRepo):
	def __init__(self) -> None:
		super().__init__()

	def __str__(self) -> str:
		return f"{self.__class__.__name__}(Sourcecode --> #{self.__len__(self.get_sourcecode())})"

	def __repr__(self) -> str:
		return self.__str__()
	
	def __len__(self, arg: Union[Sequence, Text, Dict, Set]) -> int:
		if (isinstance(arg, (int, float, bool))):
			raise TypeError("Invalid input. Only text, sequence, mapping and set are accepted")
		else:
			return len(arg)
	
	def get_nodes(self) -> List[List[Node]]:
		nodes = []
		trees, uncompiled_sourcecode = self.get_trees(self.get_sourcecode())

		if not uncompiled_sourcecode:
			for tree in trees:
				temp = []
				for _, node in tree:
					temp.append(node.__class__.__name__)
				nodes.append(temp)
		else:
			raise JavaParserError (f"Uncompiled source code --> {uncompiled_sourcecode}")

		return nodes

	def select_nodes(self) -> List[List[Node]]:
		selected = []

		for nodes in self.get_nodes():
			temp = []
			for node in nodes:
				if node in self.node_types().get("declaration"):
					temp.append(node)
				elif node in self.node_types().get("statement"):
					temp.append(node)
				elif node in self.node_types().get("expression"):
					temp.append(node)
				elif node in self.node_types().get("invocation"):
					temp.append(node)
				elif node in self.node_types().get("other"):
					temp.append(node)
			selected.append(temp)
	
		return selected 
	
	def model(self) -> models:
		tokenizer = RegexpTokenizer("\w+")
		tokens = [tokenizer.tokenize(node) for node in self.get_nodes()]
		return Word2Vec(sentences = tokens, min_count = 1, size = 32)
	
	def save_vectors(self) -> None:
		node_vecs = self.model().wv
		node_vecs.save("node_vecs.wordvectors")
	
	def assign_vectors(self) -> List[List[float]]:
		vec_dict = {}

		node_vecs = KeyedVectors.load("node_vecs.wordvectors", mmap = "r")
		for key in node_vecs.vocab:
			vec_dict[key] = node_vecs[key]
		node_vecs = []
		for node_vec in self.node_types():
			temp = []
			if node_vec is not None:
				for node in node_vec:
					if node in vec_dict:
						temp.append(vec_dict.get(node))
			node_vecs.append(temp)

		return node_vecs

	def flatten_vectors(self) -> List[float]:
		flatten_vecs = []

		for vector_list in self.assign_vectors():
			if not vector_list:
				flatten_vecs.append(vector_list)
			else:
				flatten_list = np.concatenate(vector_list).ravel().tolist()
				flatten_vecs.append(flatten_list) 

		return flatten_vecs
	
	def clean_data(self) -> Union[Tuple[List[float], int]]:
		cleaned = []

		max_length = max([self.__len__(sub_list) for sub_list in self.flatten_vectors()]) #26656
		for sublist in self.flatten_vectors(): 
			if not sublist:
				cleaned.append(np.zeros(max_length).tolist())
			elif len(sublist) < max_length:
				sublist.extend(np.zeros(max_length - self.__len__(sublist)))
				cleaned.append(sublist)
			else:
				cleaned.append(sublist)
				
		return cleaned, max_length

def build_tree_data(self) -> None:
	try:
		with open(self.FILE_PATH, "w") as file:
			with file:
				write = csv.writer(file)
				write.writerows(self.clean_data())
	except OSError as e:
		raise e
		
def transform_tree_data(self) -> int:
	max_length = self.clean_data()[1]
	scalars = [f"scalar{i}" for i in range(max_length)]
	return self.__len__(scalars)
