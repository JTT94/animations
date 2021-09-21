
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import ott

from ott.geometry import pointcloud
from ott.core import sinkhorn
from ott.tools import transport


rng = jax.random.PRNGKey(0)
rngs = jax.random.split(rng, 3)

n, m, d = 10, 10, 1
x = jax.random.normal(rngs[0], (n,d)) + 1
y = x# jax.random.uniform(rngs[1], (m,d))
x = jnp.sort(x)


a = jax.random.uniform(rngs[0], (n,))
b = jax.random.uniform(rngs[1], (m,))
a = a / jnp.sum(a)
b = b / jnp.sum(b)

geom = pointcloud.PointCloud(x, y, epsilon=1e-2)
out = sinkhorn.sinkhorn(geom, a, b)
P = geom.transport_from_potentials(out.f, out.g)

plt.imshow(P, cmap='Purples')
plt.colorbar()
plt.show()


# %%
P.T