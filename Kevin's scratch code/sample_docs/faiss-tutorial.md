---
layout: ebook-post
title: "Introduction to Facebook AI Similarity Search (Faiss)"
headline: "Introduction to Facebook AI Similarity Search (Faiss)"
categories:
  - "Faiss: The Missing Manual"
toc: >-
weight: 1
author:
  name: James Briggs
  position: Developer Advocate
  src: /images/james-briggs.jpeg
  href: "https://www.youtube.com/c/jamesbriggs"
description: Learn how Facebook AI Similarity Search changes — search.
# Open Graph
images: ['/images/faiss1.png']
---

<!-- ![Getting Started With FAISS](/images/faiss1.png) -->

Facebook AI Similarity Search (Faiss) is one of the most popular implementations of efficient similarity search, but what is it — and how can we use it?

What is it that makes [Faiss](https://github.com/facebookresearch/faiss) special? How do we make the best use of this incredible tool?

---

**Note: [Pinecone](/) lets you implement vector search into your applications with just a few API calls, without knowing anything about Faiss. However, you like seeing how things work, so enjoy the guide!**

---

Fortunately, it’s a brilliantly simple process to get started with. And in this article, we’ll explore some of the options FAISS provides, how they work, and — most importantly — how Faiss can make our search faster.

Check out the video walkthrough here:

<div style="left: 0; width: 100%; height: 0; position: relative; padding-bottom: 56.25%;">
    <iframe style="border: 1; top: 0; left: 0; width: 100%; height: 100%; position: absolute;" src="https://www.youtube-nocookie.com/embed/sKyvsdEv6rk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

## What is Faiss?

Before we get started with any code, many of you will be asking — what is Faiss?

Faiss is a library — developed by Facebook AI — that enables efficient similarity search.

So, given a set of [vectors](/learn/vector-embeddings/), we can index them using Faiss — then using another vector (the query vector), we search for the most similar vectors within the index.

Now, Faiss not only allows us to build an index and search — but it also speeds up search times to ludicrous performance levels — something we will explore throughout this article.

## Building Some Vectors

The first thing we need is data, we’ll be concatenating several datasets from this semantic test similarity hub repo. We will download each dataset, and extract the relevant text columns into a single list.

{{< notebook file="get-sentence-data" height="full" >}}

Next, we remove any duplicates, leaving us with 14.5K unique sentences. Finally, we build our dense vector representations of each sentence using the [sentence-BERT](/learn/semantic-search/) library.

{{< notebook file="create-embeddings" height="full" >}}

Now, building these sentence embeddings can take some time — so feel free to download them directly from here (you can use [this script](https://github.com/jamescalam/data/blob/main/sentence_embeddings_15K/download.py) to load them into Python).

## Plain and Simple

We’ll start simple. First, we need to set up Faiss. Now, if you’re on Linux — you’re in luck — Faiss comes with built-in GPU optimization for any CUDA-enabled Linux machine.

MacOS or Windows? Well, we’re less lucky.

_(Don’t worry, it’s still ludicrously fast)_

So, CUDA-enabled Linux users, type `conda install -c pytorch faiss-gpu`. Everyone else, `conda install -c pytorch faiss-cpu`. If you don’t want to use `conda` there are alternative installation instructions [here](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md).

Once we have Faiss installed we can open Python and build our first, plain and simple index with `IndexFlatL2`.

## IndexFlatL2

`IndexFlatL2` measures the L2 (or Euclidean) distance between _all_ given points between our query vector, and the vectors loaded into the index. It’s simple, _very_ accurate, but not too fast.

![L2 distance calculation between a query vector xq and our indexed vectors (shown as y)](/images/faiss2.png)

<small>L2 distance calculation between a query vector <b>xq</b> and our indexed vectors (shown as <b>y</b>)</small>

In Python, we would initialize our `IndexFlatL2` index with our vector dimensionality (`768` — the output size of our sentence embeddings) like so:

{{< notebook file="IndexFlatL2-init" height="full" >}}

Often, we’ll be using indexes that require us to train them before loading in our data. We can check whether an index needs to be trained using the `is_trained` method. `IndexFlatL2` is not an index that requires training, so we should return `False`.

Once ready, we load our embeddings and query like so:

{{< notebook file="IndexFlatL2-add" height="full" >}}

Which returns the top `k` vectors closest to our query vector `xq` as `7460`, `10940`, `3781`, and `5747`. Clearly, these are all great matches — all including either people running with a football or in the _context_ of a football match.

Now, if we’d rather extract the numerical vectors from Faiss, we can do that too.

{{< notebook file="reconstruct" height="full" >}}

### Speed

Using the `IndexFlatL2` index alone is computationally expensive, it doesn’t scale well.

When using this index, we are performing an _exhaustive_ search — meaning we compare our query vector `xq` to every other vector in our index, in our case that is 14.5K L2-distance calculations for every search.

Imagine the speed of our search for datasets containing 1M, 1B, or even more vectors — and when we include several query vectors?

![Milliseconds taken to return a result (y-axis) / number of vectors in the index (x-axis) — relying solely on IndexFlatL2 quickly becomes slow](/images/faiss3.png)

<small>Milliseconds taken to return a result (y-axis) / number of vectors in the index (x-axis) — relying solely on IndexFlatL2 quickly becomes slow</small>

Our index quickly becomes too slow to be useful, so we need to do something different.

## Partitioning The Index

Faiss allows us to add multiple steps that can optimize our search using many different methods. A popular approach is to partition the index into Voronoi cells.

![We can imagine our vectors as each being contained within a Voronoi cell — when we introduce a new query vector, we first measure its distance between centroids, then restrict our search scope to that centroid’s cell.](/images/faiss4.png)

<small>We can imagine our vectors as each being contained within a Voronoi cell — when we introduce a new query vector, we first measure its distance between centroids, then restrict our search scope to that centroid’s cell.</small>

Using this method, we would take a query vector `xq`, identify the cell it belongs to, and then use our `IndexFlatL2` (or another metric) to search between the query vector and all other vectors belonging to _that specific_ cell.

So, we are reducing the scope of our search, producing an _approximate_ answer, rather than exact (as produced through exhaustive search).

To implement this, we first initialize our index using `IndexFlatL2` — but this time, we are using the L2 index as a quantizer step — which we feed into the partitioning `IndexIVFFlat` index.

{{< notebook file="IndexIVFFlat-init" height="full" >}}

Here we’ve added a new parameter `nlist`. We use `nlist` to specify how many partitions (Voronoi cells) we’d like our index to have.

Now, when we built the previous `IndexFlatL2`-only index, we didn’t need to train the index as no grouping/transformations were required to build the index. Because we added clustering with `IndexIVFFlat`, this is no longer the case.

So, what we do now is train our index on our data — which we must do _before_ adding any data to the index.

{{< notebook file="IndexIVFFlat-train" height="full" >}}

Now that our index is trained, we add our data just as we did before.

Let’s search again using the same indexed sentence embeddings and the same query vector `xq`.

{{< notebook file="IndexIVFFlat-search" height="full" >}}

The search time has clearly decreased, in this case, we don’t find any difference between results returned by our exhaustive search, and this approximate search. But, often this can be the case.

If approximate search with `IndexIVFFlat` returns suboptimal results, we can improve accuracy by increasing the search scope. We do this by increasing the `nprobe` attribute value — which defines how many nearby cells to search.

![Searching the single closest cell when nprobe == 1 (left), and searching the eight closest cells when nprobe == 8 (right)](/images/faiss5.png)

<small>Searching the single closest cell when <b>nprobe == 1</b> (left), and searching the eight closest cells when <b>nprobe == 8</b> (right)</small>

We can implement this change easily.

{{< notebook file="IndexIVFFlat-nprobe" height="full" >}}

Now, because we’re searching a larger scope by increasing the `nprobe` value, we will see the search speed increase too.

![Query time / number of vectors for the IVFFlat index with different nprobe values — 1, 5, 10, and 20](/images/faiss6.png)

<small>Query time / number of vectors for the IVFFlat index with different <b>nprobe</b> values — 1, 5, 10, and 20</small>

Although, even with the larger `nprobe` value we still see much faster responses than we returned with our `IndexFlatL2`-only index.

### Vector Reconstruction

If we go ahead and attempt to use `index.reconstruct(<vector_idx>)` again, we will return a `RuntimeError` as there is no direct mapping between the original vectors and their index position, due to the addition of the IVF step.

So, if we’d like to reconstruct the vectors, we must first create these direct mappings using `index.make_direct_map()`.

{{< notebook file="make-direct-map" height="full" >}}

And from there we are able to reconstruct our vectors just as we did before.

## Quantization

We have one more key optimization to cover. All of our indexes so far have stored our vectors as full (eg `Flat`) vectors. Now, in very large datasets this can quickly become a problem.

Fortunately, Faiss comes with the ability to compress our vectors using _Product Quantization (PQ)_.

But, what is PQ? Well, we can view it as an additional approximation step with a similar outcome to our use of **IVF**. Where IVF allowed us to approximate by _reducing the scope_ of our search, PQ approximates the _distance/similarity calculation_ instead.

PQ achieves this approximated similarity operation by compressing the vectors themselves, which consists of three steps.

![Three steps of product quantization](/images/faiss7.png)

<small>Three steps of product quantization</small>

1. We split the original vector into several subvectors.
2. For each set of subvectors, we perform a clustering operation — creating multiple centroids for each sub-vector set.
3. In our vector of sub-vectors, we replace each sub-vector with the ID of it’s nearest set-specific centroid.

To implement all of this, we use the IndexIVF**PQ** index — we’ll also need to `train` the index before adding our embeddings.

{{< notebook file="IndexIVFPQ-init" height="full" >}}

And now we’re ready to begin searching using our new index.

{{< notebook file="IndexIVFPQ-search" height="full" >}}

### Speed or Accuracy?

Through adding PQ we’ve reduced our IVF search time from ~7.5ms to ~5ms, a small difference on a dataset of this size — but when scaled up this becomes significant quickly.

However, we should also take note of the slightly different results being returned. Beforehand, with our exhaustive L2 search, we were returning `7460`, `10940`, `3781`, and `5747`. Now, we see a slightly different order of results — and two different IDs, `5013` and `5370`.

Both of our speed optimization operations, **IVF** and **PQ**, come at the cost of accuracy. Now, if we print out these results we will still find that each item is relevant:

{{< notebook file="IndexIVFPQ-results" height="full" >}}

So, although we might not get the _perfect_ result, we still get close — and thanks to the approximations, we get a much faster response.

![Query time / number of vectors for our three indexes](/images/faiss8.png)

<small>Query time / number of vectors for our three indexes</small>

And, as shown in the graph above, the difference in query times become increasingly relevant as our index size increases.

That’s it for this article! We’ve covered the essentials to getting started with building high-performance indexes for search in Faiss.

Clearly, a lot can be done using `IndexFlatL2`, `IndexIVFFlat`, and `IndexIVFPQ` — and each has many parameters that can be fine-tuned to our specific accuracy/speed requirements. And as shown, we can produce some truly impressive results, at lightning-fast speeds very easily thanks to Faiss.

---

**Want to run Faiss in production? [Pinecone](/) provides vector similarity search that's production-ready, scalable, and fully managed.**