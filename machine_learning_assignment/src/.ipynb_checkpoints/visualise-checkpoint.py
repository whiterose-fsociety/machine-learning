import pydot
# Credit
# https://stackoverflow.com/questions/13688410/dictionary-object-to-decision-tree-in-pydot

graph = pydot.Dot(graph_type="graph")

def draw(parent_name, child_name):
    edge = pydot.Edge(parent_name, child_name)
    # graph.add_edge(edge)


def visit(node, parent=None):
    for k,v in node.items():
        if isinstance(v, dict):      
            if parent:
                draw(parent, k)
            visit(v, k)
        else:
            draw(parent, k)         
            draw(k, k+'_'+v)


def visualise(model):
	visit(model)
	graph.write_png("decision_tree.png")