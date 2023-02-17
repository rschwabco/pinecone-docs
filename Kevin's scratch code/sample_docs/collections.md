---
title: Collections
category: 630fc5235d91a70054705fb8
---

## Overview

This document explains the concepts related to collections in Pinecone.

> ⚠️  Warning
>
> This is a **public preview** ("Beta") feature. Test thoroughly before
> using this feature for production workloads. No SLAs or technical support
> commitments are provided for this feature.

**A collection is a static copy of an index.** It is a non-queryable representation of a set of vectors and metadata. You can create a collection from an index, and you can create a new index from a collection. This new index can differ from the original source index: the new index can have a different number of pods, a different pod type, or a different similarity metric.

## Use cases for collections

Creating a collection from your index is useful when performing tasks like the following:

+ Temporarily shutting down an index
+ Copying the data from one index into a different index;
+ Making a backup of your index
+ Experimenting with different index configurations

To learn about creating backups with collections, see [Back up indexes](back-up-indexes/#create-a-backup-using-a-collection).

To learn about creating indexes from collections, see [Manage indexes](manage-indexes/#create-an-index-from-a-collection).


## Public collections contain real world data

Public collections contain vectorized data from real-world datasets that you can use to [create
indexes](manage-indexes/#create-an-index-from-a-public-collection). You can use these indexes to try out Pinecone with realistic example data and queries. 

Pinecone offers public collections containing data from the following datasets:

+ [OpenAI TREC](https://huggingface.co/datasets/trec)
+ [Cohere TREC](https://huggingface.co/datasets/trec)
+ [SQuAD](https://huggingface.co/datasets/squad)

## Performance 

Collections operations perform differently with different pod types.

+ Creating a collection from an index takes approximately 10 minutes. 
+ Creating a p1 or s1 index from a collection takes approximately 10 minutes.
+ Creating a p2 index from a collection can take several hours.

## Limitations

You cannot query or write to a collection after its creation. For this reason, a collection only incurs storage costs.

You can only perform operations on collections in the current Pinecone project.
