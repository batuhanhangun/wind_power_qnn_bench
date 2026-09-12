"""Guard the verified experiment bodies against accidental numerical rewrites."""
import ast
import hashlib
import json
from qnnbench.config import ROOT

class WithoutDocstrings(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
            node.body.pop(0)
        return node

def test_original_experiment_function_bodies():
    records=json.loads((ROOT/'tests/original/function_fingerprints.json').read_text(encoding='utf-8'))
    for r in records:
        tree=WithoutDocstrings().visit(ast.parse((ROOT/r['destination']).read_text(encoding='utf-8')))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==r['function'])
        digest=hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest()
        assert digest==r['ast_sha256'],f"Original function changed: {r['source']}:{r['function']}"
