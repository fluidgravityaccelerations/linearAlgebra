# Linear Algebra

Implementations of the following linear algebra routines:

- **LU Decomposition by Crout’s Method** (in Matlab and Python)
- **Solution of Linear Systems using LU Decomposition** (in Matlab and Python)

The codes are designed for educational purposes only. 

---

## 📐 Mathematical Background

Given a linear system:

$$Ax=b$$

the goal is to compute the unknown vector $x$ using an efficient and numerically stable method.
One of the most common techniques is LU decomposition, which expresses the matrix $A$ as the product of a lower triangular matrix $L$ and an upper triangular matrix $U$:

$$A=LU$$

---

## Crout’s LU Decomposition

Crout’s method constructs the matrices $L$ and $U$ such that:

$L$ is lower triangular with non-unit diagonal elements.

$U$ is upper triangular with unit diagonal elements.

For an $n\times n$ matrix $A$, the decomposition satisfies:

$$A_{ij}​=\sum_{k=1}​^{\min(i,j)}​​L_{ik}​​U_{kj}​.$$
