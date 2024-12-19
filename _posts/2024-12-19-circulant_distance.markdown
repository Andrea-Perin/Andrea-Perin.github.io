---
layout: distill
title: Riemannian distance for SPD circulant matrices
description: Or slicing conees
tags: maths
giscus_comments: false
date: 2024-12-19
featured: false
---

I was working on some homework for a course on *Geometric Deep Learning*.
The context was differential geometry, more specifically the manifold of Symmetric Positive Definite (SPD) matrices.
The exercise had us thinking about a notion of distance for this manifold.
It also was presenting it as an example of the algebraic concept of a [convex cone](https://en.wikipedia.org/wiki/Convex_cone).
I was left wondering: where do *circulant matrices* live in all of this?

## Preliminaries
Some context.

##### SPD matrices
If you end up on this page, I take it for granted you know what a symmetric matrix is.
As for the Positive Definite part, the way I think about them is like a "stretching" operator: feed it any vector, and this thing will:
1. rotate it to "align it" to an orthogonal basis;
2. stretch/scale every component by a strictly positive number;
3. undo the rotation you did at the beginning.
Importantly, an positive definite matrix has strictly positive determinant.

More info is [on Wikipedia](https://en.wikipedia.org/wiki/Definite_matrix), of course.

You may find SPD matrices in the following contexts:
* as *covariance matrices*;
* as *metric tensors*, once you specify some coordinate system;
* as *sample covariance matrices*, as long as there are "enough samples";
* as *Gram matrices*, in the context of kernel methods;
* and, I assume, many more.

##### The manifold of SPD matrices
Imagine an $n\times n$ matrix $\cal M$ that is symmetric and positive definite.
Due to its symmetry, you only actually need to specify $n(n+1)/2$ elements to characterize the matrix.
You can then think of this matrix as a point in the space $\mathbb{R}^{n(n+1)/2}$.
The set of all such points that correspond to a symmetric positive definite matrix form a *manifold*.
It is possible to endow this manifold with a *metric*, that is, a way to "measure distance" between two of its points.
A careful discussion of this is a bit outside of my capabilities, but resources are plentiful if you crave some big boy maths.

It is instructive and amusing to check what this manifold may look like in practice.
Choose $n=2$, so that matrices (which are symmetric!) are of the type
$$
\cal M = \begin{pmatrix}
    a & c \\
    c & b
    \end{pmatrix}.
$$
Additionally, we need to satisfy the requirements $\det \cal M = ab - c^2 >0$, and $a>0, b>0$ (this is because of the positive-definiteness requirement).
The matrix $\cal M$ can be represented as the point $(a, b, c)\in \mathbb{R}^3$.
The manifold is then the subset of $\mathbb{R}^3$ which satisfies these requirements:
* $ab -c^2>0$;
* $a>0$;
* $b>0$.

Feed this to `matplotlib`, and you get this: ![The convex cone structure for a 2D SPD matrix.](/assets/img/posts/circulant_distance/cone.png)
