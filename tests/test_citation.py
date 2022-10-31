from src.environments.citation_wrapper import CitationDynamics
import numpy as np

citation = CitationDynamics()

citation.initialize()

print(citation.x)
print(citation.t)


citation.step(u=np.random.rand(3))
print(citation.x)
print(citation.t)

citation.step(u=np.random.rand(3))
print(citation.x)
print(citation.t)
