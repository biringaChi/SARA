import os
import javalang
import pandas as pd
from pandas.core.frame import DataFrame
from javalang.tree import CompilationUnit
from javalang.parser import JavaSyntaxError
from typing import Tuple, Dict, List, Sequence, Set, Text, Union, TypeVar

Sourcecode = TypeVar("Sourcecode")
FeatureVec = TypeVar("FeatureVec")

class HandleCodeRepo:
	@property
	def _GET_DATA_DIR(self) -> List[str]:
		datasets = {"AddressBook" : "addressbook.csv", "DSpace" : "dspace.csv"} 
		return os.path.join("./data", datasets.get("AddressBook")) # select dataset

	def __str__(self) -> str:
		return f"{self.__class__.__name__}({self._GET_DATA_DIR})"

	def __repr__(self) -> str:
		return self.__str__()
	
	def __len__(self, arg: Union[Sequence, Text, Dict, Set]) -> int:
		if (isinstance(arg, (int, float, bool))):
			raise TypeError("Invalid argument. Only text, sequence, mapping and set are accepted")
		else:
			return len(arg)
	
	def node_types(self) -> Union[Dict[str, List[str]], str, List[str]]:
		return {
			"declaration" : {
				"TypeDeclaration", "FieldDeclaration", "MethodDeclaration", 
				"ConstructorDeclaration", "PackageDeclaration", "ClassDeclaration", 
				"EnumDeclaration", "InterfaceDeclaration", "AnnotationDeclaration", 
				"ConstantDeclaration", "VariableDeclaration", "LocalVariableDeclaration",
				"EnumConstantDeclaration"
				},
			"statement" : {
				"IfStatement", "WhileStatement", "DoStatement",
				"AssertStatement", "SwitchStatement", "ForStatement",
				"ContinueStatement", "ReturnStatement", "ThrowStatement",
				"SynchronizedStatement", "TryStatement", "BreakStatement",
				"BlockStatement"
				},
			"expression" : {
				"StatementExpression", "TernaryExpression", "LambdaExpression"
				},
			"invocation" : {
				"SuperConstructorInvocation", "MethodInvocation",  "SuperMethodInvocation",
				"ExplicitConstructorInvocation"
				},
			"other" : {
				"ForControl", "EnhancedForControl", "FormalParameter",
				"AnnotationMethod", "VariableDeclarator", "CatchClause",
				"BinaryOperation", "MethodReference", "ArraySelector",
				"SuperMemberReference"
				},
			}

	def read_data(self) -> DataFrame:
		try:
			return pd.read_csv(self._GET_DATA_DIR)
		except FileNotFoundError as e:
			raise(e)
	
	def get_hash(self) -> List[str]:
		return [hash for hash in self.read_data()["hash"]]

	def get_sourcecode(self) -> Sourcecode:
		return [code for code in self.read_data()["sourcecode"]]

	def get_unit_test(self) -> List[float]:
		return [code for code in self.read_data()["unit_test"]] 

	def get_integration_test(self) -> List[float]:
		return [code for code in self.read_data()["integration_test"]]

	def unit_integration_test(self) -> List[float]:
		return [code for code in self.read_data()["unit_integration_test"]]

	def get_trees(self, code) -> Union[Tuple[List[CompilationUnit], Dict[int, str]]]:
		trees = []
		uncompiled_sourcecode = {}

		for idx, sourcecode in enumerate(code):
			try:
				trees.append(javalang.parse.parse(sourcecode))
			except JavaSyntaxError as e:
				uncompiled_sourcecode[idx] = sourcecode
				trees.append(None)

		return trees, uncompiled_sourcecode
		