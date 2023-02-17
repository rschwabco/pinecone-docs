---
title: GIF Search
category: 630fc5235d91a70054705fb7
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pinecone-io/examples/blob/master/search/semantic-search/gif-search/gif-search.ipynb) [![Open nbviewer](https://raw.githubusercontent.com/pinecone-io/examples/master/assets/nbviewer-shield.svg)](https://nbviewer.org/github/pinecone-io/examples/blob/master/search/semantic-search/gif-search/gif-search.ipynb) [![Open Github](https://raw.githubusercontent.com/pinecone-io/examples/master/assets/github-shield.svg)](https://github.com/pinecone-io/examples/tree/master/search/semantic-search/gif-search/gif-search.ipynb)

We will use the [Tumblr GIF Description Dataset](http://raingo.github.io/TGIF-Release/), which contains over 100k animated GIFs and 120K sentences describing its visual content. Using this data with a *vector database* and *retriever* we are able to create an NLP-powered GIF search tool.

There are a few packages that must be installed for this notebook to run:


```python
pip install -U pandas pinecone-client sentence-transformers tqdm
```

We must also set the following notebook parameters to display the GIF images we will be working with.


```python
from IPython.display import HTML
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```

## Download and Extract Dataset

First let's download and extract the dataset. The dataset is available [here](https://github.com/raingo/TGIF-Release) on GitHub. We can use the link below to download the dataset directly. We can also access the link from a browser to directly download the files.


```python
# Use wget to download the master.zip file which contains the dataset
!wget https://github.com/raingo/TGIF-Release/archive/master.zip
```


```python
# Use unzip to extract the master.zip file
!unzip master.zip
```

## Explore the Dataset

Now let's explore the downloaded files. The data we want is in *tgif-v1.0.tsv* file in the *data* folder. We can use *pandas* library to open the file. We need to set delimiter as `\t` as the file contains tab separated values.


```python
import pandas as pd
```


```python
# Load dataset to a pandas dataframe
df = pd.read_csv(
    "./TGIF-Release-master/data/tgif-v1.0.tsv",
    delimiter="\t",
    names=['url', 'description']
)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://38.media.tumblr.com/9f6c25cc350f12aa74...</td>
      <td>a man is glaring, and someone with sunglasses ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://38.media.tumblr.com/9ead028ef62004ef6a...</td>
      <td>a cat tries to catch a mouse on a tablet</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://38.media.tumblr.com/9f43dc410be85b1159...</td>
      <td>a man dressed in red is dancing.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://38.media.tumblr.com/9f659499c8754e40cf...</td>
      <td>an animal comes close to another in the jungle</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://38.media.tumblr.com/9ed1c99afa7d714118...</td>
      <td>a man in a hat adjusts his tie and makes a wei...</td>
    </tr>
  </tbody>
</table>
</div>



*Note the dataset does not contain the actual GIF files. But it has URLs we can use to download/access the GIF files. This is great as we do not need to store/download all the GIF files. We can directly load the required GIF files using the URL when displaying the search results.*

There are some duplicate descriptions in the dataset.


```python
len(df)
```




    125782




```python
# Number of *unique* GIFs in the dataset
len(df["url"].unique())
```




    102068




```python
dupes = df['url'].value_counts().sort_values(ascending=False)
dupes.head()
```




    https://38.media.tumblr.com/ddbfe51aff57fd8446f49546bc027bd7/tumblr_nowv0v6oWj1uwbrato1_500.gif    4
    https://33.media.tumblr.com/46c873a60bb8bd97bdc253b826d1d7a1/tumblr_nh7vnlXEvL1u6fg3no1_500.gif    4
    https://38.media.tumblr.com/b544f3c87cbf26462dc267740bb1c842/tumblr_n98uooxl0K1thiyb6o1_250.gif    4
    https://33.media.tumblr.com/88235b43b48e9823eeb3e7890f3d46ef/tumblr_nkg5leY4e21sof15vo1_500.gif    4
    https://31.media.tumblr.com/69bca8520e1f03b4148dde2ac78469ec/tumblr_npvi0kW4OD1urqm0mo1_400.gif    4
    Name: url, dtype: int64



Let's take a look at one of these duplicated URLs and it's descriptions.


```python
dupe_url = "https://33.media.tumblr.com/88235b43b48e9823eeb3e7890f3d46ef/tumblr_nkg5leY4e21sof15vo1_500.gif"
dupe_df = df[df['url'] == dupe_url]

# let's take a look at this GIF and it's duplicated descriptions
for _, gif in dupe_df.iterrows():
    HTML(f"<img src={gif['url']} style='width:120px; height:90px'>")
    print(gif["description"])
```




<img src=https://33.media.tumblr.com/88235b43b48e9823eeb3e7890f3d46ef/tumblr_nkg5leY4e21sof15vo1_500.gif style='width:120px; height:90px'>



    two girls are singing music pop in a concert





<img src=https://33.media.tumblr.com/88235b43b48e9823eeb3e7890f3d46ef/tumblr_nkg5leY4e21sof15vo1_500.gif style='width:120px; height:90px'>



    a woman sings sang girl on a stage singing





<img src=https://33.media.tumblr.com/88235b43b48e9823eeb3e7890f3d46ef/tumblr_nkg5leY4e21sof15vo1_500.gif style='width:120px; height:90px'>



    two girls on a stage sing into microphones.





<img src=https://33.media.tumblr.com/88235b43b48e9823eeb3e7890f3d46ef/tumblr_nkg5leY4e21sof15vo1_500.gif style='width:120px; height:90px'>



    two girls dressed in black are singing.


There is no reason for us to remove these duplicates, as shown here, every description is accurate. You can spot check a few of the other URLs but they all seem to be the same where we have several *accurate* descriptions for a single GIF.

That leaves us with 125,781 descriptions for 102,067 GIFs. We will use these descriptions to create *context* vectors that will be indexed in a vector database to create our GIF search tool. Let's take a look at a few more examples of GIFs and their descriptions.


```python
for _, gif in df[:5].iterrows():
  HTML(f"<img src={gif['url']} style='width:120px; height:90px'>")
  print(gif["description"])
```




<img src=https://38.media.tumblr.com/9f6c25cc350f12aa74a7dc386a5c4985/tumblr_mevmyaKtDf1rgvhr8o1_500.gif style='width:120px; height:90px'>



    a man is glaring, and someone with sunglasses appears.





<img src=https://38.media.tumblr.com/9ead028ef62004ef6ac2b92e52edd210/tumblr_nok4eeONTv1s2yegdo1_400.gif style='width:120px; height:90px'>



    a cat tries to catch a mouse on a tablet





<img src=https://38.media.tumblr.com/9f43dc410be85b1159d1f42663d811d7/tumblr_mllh01J96X1s9npefo1_250.gif style='width:120px; height:90px'>



    a man dressed in red is dancing.





<img src=https://38.media.tumblr.com/9f659499c8754e40cf3f7ac21d08dae6/tumblr_nqlr0rn8ox1r2r0koo1_400.gif style='width:120px; height:90px'>



    an animal comes close to another in the jungle





<img src=https://38.media.tumblr.com/9ed1c99afa7d71411884101cb054f35f/tumblr_mvtuwlhSkE1qbnleeo1_500.gif style='width:120px; height:90px'>



    a man in a hat adjusts his tie and makes a weird face.


We can see that the description of the GIF accurately describes what is happening in the GIF, we can use these descriptions to search through our GIFs.

Using this data, we can build the GIF search tool with just *two* components:

* a **retriever** to embed GIF descriptions
* a **vector database** to store GIF description embeddings and retrieve relevant GIFs

## Initialize Pinecone Index

The vector database stores vector representations of our GIF descriptions which we can retrieve using another vector (query vector). We will use the Pinecone vector database, a fully managed vector database that can store and search through billions of records in milliseconds. You could use any other vector database such as FAISS to build this tool. But you may need to manage the database yourself.

To initialize the database, we sign up for a [free Pinecone API key](https://app.pinecone.io/) and `pip install pinecone-client`. You can find your environment in the [Pinecone console](https://app.pinecone.io) under **API Keys**. Once ready, we initialize our index with:


```python
import pinecone

# Connect to pinecone environment
pinecone.init(
    api_key="<<YOUR_API_KEY>>",
    environment="YOUR_ENVIRONMENT"
)

index_name = 'gif-search'

# check if the gif-search exists
if index_name not in pinecone.list_indexes():
    # create the index if it does not exist
    pinecone.create_index(
        index_name,
        dimension=384,
        metric="cosine"
    )

# Connect to gif-search index we created
index = pinecone.Index(index_name)
```

Here we specify the name of the index where we will store our GIF descriptions and their URLs, the similarity metric, and the embedding dimension of the vectors. The similarity metric and embedding dimension can change depending on the embedding model used. However, most retrievers use "cosine" and 768.

## Initialize Retriever

Next, we need to initialize our retriever. The retriever will mainly do two things:

1.	Generate embeddings for all the GIF descriptions (context vectors/embeddings)
2.	Generate embeddings for the query (query vector/embedding)

The retriever will generate the embeddings in a way that the queries and GIF descriptions with similar meanings are in a similar vector space. Then we can use cosine similarity to calculate this similarity between the query and context embeddings and find the most relevant GIF to our query.

We will use a `SentenceTransformer` model trained based on Microsoft's MPNet as our retriever. This model performs well out-of-the-box when searching based on generic semantic similarity. 


```python
from sentence_transformers import SentenceTransformer
```


```python
# Initialize retriever with SentenceTransformer model 
retriever = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
retriever
```




    SentenceTransformer(
      (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
      (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False})
      (2): Normalize()
    )



## Generate Embeddings and Upsert

Now our retriever and the pinecone index are initialized. Next, we need to generate embeddings for the GIF descriptions. We will do this in batches to help us more quickly generate embeddings. This means our retriever will generate embeddings for 64 GIF descriptions at once instead of generating them individually (much faster) and send a single API call for each batch of 64 (also much faster).

When passing the documents to pinecone, we need an id (a unique value), embedding (embeddings for the GIF descriptions we have generated earlier), and metadata for each document representing GIFs in the dataset. The metadata is a dictionary containing data relevant to our embeddings. For the GIF search tool, we only need the URL and description.


```python
from tqdm.auto import tqdm

# we will use batches of 64
batch_size = 64

for i in tqdm(range(0, len(df), batch_size)):
    # find end of batch
    i_end = min(i+batch_size, len(df))
    # extract batch
    batch = df.iloc[i:i_end]
    # generate embeddings for batch
    emb = retriever.encode(batch['description'].tolist()).tolist()
    # get metadata
    meta = batch.to_dict(orient='records')
    # create IDs
    ids = [f"{idx}" for idx in range(i, i_end)]
    # add all to upsert list
    to_upsert = list(zip(ids, emb, meta))
    # upsert/insert these records to pinecone
    _ = index.upsert(vectors=to_upsert)

    
# check that we have all vectors in index
index.describe_index_stats()
```


      0%|          | 0/1966 [00:00<?, ?it/s]





    {'dimension': 384,
     'index_fullness': 0.05,
     'namespaces': {'': {'vector_count': 125782}}}



We can see all our documents are now in the pinecone index. Let's run some queries to test our GIF search tool.

## Querying

We have two functions, `search_gif`, to handle our search query, and `display_gif`, to display the search results.

The `search_gif` function generates vector embedding for the search query using the retriever model and then runs the query on the pinecone index. `index.query` will compute the cosine similarity between the query embedding and the GIF description embeddings as we set the metric type as "cosine" when we initialize the pinecone index. The function will return the URL of the top 10 most relevant GIFs to our search query.


```python
def search_gif(query):
    # Generate embeddings for the query
    xq = retriever.encode(query).tolist()
    # Compute cosine similarity between query and embeddings vectors and return top 10 URls
    xc = index.query(xq, top_k=10,
                    include_metadata=True)
    result = []
    for context in xc['matches']:
        url = context['metadata']['url']
        result.append(url)
    return result
```

The `display_gif` can display multiple GIFs using its URLs in the jupyter notebook in a grid style. We use this function to display the top 10 GIFs returned by the `search_gif` function.


```python
def display_gif(urls):
    figures = []
    for url in urls:
        figures.append(f'''
            <figure style="margin: 5px !important;">
              <img src="{url}" style="width: 120px; height: 90px" >
            </figure>
        ''')
    return HTML(data=f'''
        <div style="display: flex; flex-flow: row wrap; text-align: center;">
        {''.join(figures)}
        </div>
    ''')
```

Let's begin testing some queries.


```python
gifs = search_gif("a dog being confused")
display_gif(gifs)
```





<div style="display: flex; flex-flow: row wrap; text-align: center;">

<figure style="margin: 5px !important;">
  <img src="https://38.media.tumblr.com/af53df8d946bbca23be97691db0ecd5e/tumblr_nq3l305zdF1s71nvbo1_400.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://33.media.tumblr.com/a574ab035e7edc7708db423ee67f3ac4/tumblr_nq1zodZJNx1uoke7ao1_400.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://38.media.tumblr.com/94703ea885174ffc97c44d57487d7ee9/tumblr_na6oo2PKSC1silsr6o1_400.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://38.media.tumblr.com/fa6a31e326066bb27776066150c8c810/tumblr_np38ipgJPd1tkkgpso1_250.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://38.media.tumblr.com/241d89939a5714c2db4566d9108245fe/tumblr_n9xv6aqQ5A1qmgppeo1_250.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://31.media.tumblr.com/a00ae69f826dbe89a5bdabad567ac88d/tumblr_n8x5e6ZcFW1sjpl9lo1_400.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://31.media.tumblr.com/28a9aac3c21941e1c61dd9ab4390c3f5/tumblr_nhdr3clKDa1sntw1mo1_400.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://31.media.tumblr.com/5cbd531e1d8cc7fefffdb8a68ec62b1d/tumblr_naysx8YTzn1tzl1owo1_400.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://38.media.tumblr.com/bc300fcbae8e4eb65c3901a246f46e4c/tumblr_niu5dzNP7G1u62tooo1_400.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://38.media.tumblr.com/1c3edb33951b52020b9271185942b2b2/tumblr_nflm4phy0P1u4txqeo1_250.gif" style="width: 120px; height: 90px" >
</figure>

</div>





```python
gifs = search_gif("animals being cute")
display_gif(gifs)
```





<div style="display: flex; flex-flow: row wrap; text-align: center;">

<figure style="margin: 5px !important;">
  <img src="https://33.media.tumblr.com/73841eb3b37ad5277b324359a83bb19e/tumblr_ngnz25VQpD1twctp1o1_400.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://38.media.tumblr.com/7b8ebff7051b8a7d0502294465559861/tumblr_na8n60gCmT1tiamw8o1_500.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://38.media.tumblr.com/49223a5564c8d7dfafe115063ba88c8a/tumblr_nnrps82EvG1sxvevjo1_400.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://31.media.tumblr.com/aa9c98f92f06cc3484ae395194db6d7f/tumblr_naeyc6yQWy1tahfdeo1_250.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://38.media.tumblr.com/e7a1d7ed5f2289db13e1812a91c0eedf/tumblr_nf8r2nWajt1s236zjo1_400.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://38.media.tumblr.com/7d8f9cac33b4fc76908a37bf28ab6fca/tumblr_noswtqDMKC1tyncywo1_400.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://38.media.tumblr.com/136d1d103edf3a82c2332bf8ef28d6d3/tumblr_nhm8rleyTk1u333yco2_400.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://33.media.tumblr.com/0f97de4f3cc8dca408ca4ab036460412/tumblr_njmp6tj53K1thqmhto1_400.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://31.media.tumblr.com/be2b34de9ff751da15cbde3144d25007/tumblr_nh4oj86LJO1slj978o1_400.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://38.media.tumblr.com/5f45e9a56121b070ddceca58b37e9ace/tumblr_njaggwVmdn1un7vpco1_400.gif" style="width: 120px; height: 90px" >
</figure>

</div>



```python
gifs = search_gif("an animal dancing")
display_gif(gifs)
```





<div style="display: flex; flex-flow: row wrap; text-align: center;">

<figure style="margin: 5px !important;">
  <img src="https://33.media.tumblr.com/7ada83ae354be1d83ea4407fea789ab8/tumblr_na0e6razjV1s71nvbo1_250.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://33.media.tumblr.com/3be31f4531ed041ff9b80465b56d810e/tumblr_nr0dycLuRO1useffdo1_400.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://33.media.tumblr.com/f0edc38b8dacce783bebcdf41db55a93/tumblr_npapb2c4Wz1uolkubo1_250.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://33.media.tumblr.com/2bf0f300d9ecfbcedf2dd3ba2b40b5e5/tumblr_ne5p2oTuCj1tdmffyo1_250.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://38.media.tumblr.com/3e1f37fea789bb1508d40e8c30f791ae/tumblr_na3xfcUdnK1tiamx1o1_400.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://38.media.tumblr.com/0b04187cb51a8889b0f41e5fbe390df2/tumblr_nbcltiDvcq1s7ri4yo1_400.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://38.media.tumblr.com/e52a1d77dc0a679840a715c02035e5da/tumblr_nfbeo6Qqr91tl8fnfo1_400.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://38.media.tumblr.com/a67f2f007b9881080aa3fe3584847bc5/tumblr_nc1wzyMaJP1tzj4j8o1_250.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://38.media.tumblr.com/61e9abf3681eeacea18dae288f084d62/tumblr_nbw9gwXM0e1tk2ngvo1_400.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://31.media.tumblr.com/78355496a2ed41f0aa9fe855f9460bc3/tumblr_nais3s7sWa1s3att3o1_400.gif" style="width: 120px; height: 90px" >
</figure>

</div>




Let's describe the third GIF with the ginger dog dancing on his hind legs.


```python
gifs = search_gif("a fluffy dog being cute and dancing like a person")
display_gif(gifs)
```





<div style="display: flex; flex-flow: row wrap; text-align: center;">

<figure style="margin: 5px !important;">
  <img src="https://38.media.tumblr.com/a67f2f007b9881080aa3fe3584847bc5/tumblr_nc1wzyMaJP1tzj4j8o1_250.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://33.media.tumblr.com/2bf0f300d9ecfbcedf2dd3ba2b40b5e5/tumblr_ne5p2oTuCj1tdmffyo1_250.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://33.media.tumblr.com/ec768e8a6f881fbc0f329932c8591a88/tumblr_mpqwb14Fsq1rjcfxro1_250.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://33.media.tumblr.com/7ada83ae354be1d83ea4407fea789ab8/tumblr_na0e6razjV1s71nvbo1_250.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://38.media.tumblr.com/f8b6d3d79b59462019c2daf2ba8b4148/tumblr_np762bxBYV1t7jda2o1_400.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://38.media.tumblr.com/a5ae79c2d62c592d7565684a72af8f2c/tumblr_nageslBNqC1tstoffo1_500.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://38.media.tumblr.com/3e1f37fea789bb1508d40e8c30f791ae/tumblr_na3xfcUdnK1tiamx1o1_400.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://33.media.tumblr.com/aec9cbbdf826f98307e6d5f3d544a4c2/tumblr_mmlrbhGDAO1qaqutao1_500.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://38.media.tumblr.com/0b04187cb51a8889b0f41e5fbe390df2/tumblr_nbcltiDvcq1s7ri4yo1_400.gif" style="width: 120px; height: 90px" >
</figure>

<figure style="margin: 5px !important;">
  <img src="https://33.media.tumblr.com/14f9b213a7355096c14b0af3a7768f5d/tumblr_npexfuFU2K1ti77bgo1_400.gif" style="width: 120px; height: 90px" >
</figure>

</div>




These look like pretty good, interesting results.

## Example application

To try out an application like this one, see this [example
application](https://huggingface.co/spaces/pinecone/gif-search).


---
