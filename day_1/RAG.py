# pip install langchain langchain-community chromadb

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings




document= """ India, officially the Republic of India,[j][20] is a country in South Asia. It is the seventh-largest country by area; the most populous country since 2023;[21] and, since its independence in 1947, the world's most populous democracy.[22][23][24] Bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast, it shares land borders with Pakistan to the west;[k] China, Nepal, and Bhutan to the north; and Bangladesh and Myanmar to the east. In the Indian Ocean, India is near Sri Lanka and the Maldives; its Andaman and Nicobar Islands share a maritime border with Myanmar, Thailand, and Indonesia.

Modern humans arrived on the Indian subcontinent from Africa no later than 55,000 years ago.[26][27][28] Their long occupation, predominantly in isolation as hunter-gatherers, has made the region highly diverse.[29] Settled life emerged on the subcontinent in the western margins of the Indus river basin 9,000 years ago, evolving gradually into the Indus Valley Civilisation of the third millennium BCE.[30] By 1200 BCE, an archaic form of Sanskrit, an Indo-European language, had diffused into India from the northwest.[31][32] Its hymns recorded the early dawnings of Hinduism in India.[33] India's pre-existing Dravidian languages were supplanted in the northern regions.[34] By 400 BCE, caste had emerged within Hinduism,[35] and Buddhism and Jainism had arisen, proclaiming social orders unlinked to heredity.[36] Early political consolidations gave rise to the loose-knit Maurya and Gupta Empires.[37] Widespread creativity suffused this era,[38] but the status of women declined,[39] and untouchability became an organised belief.[l][40] In South India, the Middle kingdoms exported Dravidian language scripts and religious cultures to the kingdoms of Southeast Asia.[41]

In the 1st millennium, Christianity, Islam, Judaism, and Zoroastrianism became established on India's southern and western coasts.[42] In the early centuries of the 2nd millennium Muslim armies from Central Asia intermittently overran India's northern plains.[43] The resulting Delhi Sultanate drew northern India into the cosmopolitan networks of medieval Islam.[44] In south India, the Vijayanagara Empire created a long-lasting composite Hindu culture.[45] In the Punjab, Sikhism emerged, rejecting institutionalised religion.[46] The Mughal Empire ushered in two centuries of economic expansion and relative peace,[47] and left a a rich architectural legacy.[48][49] Gradually expanding rule of the British East India Company turned India into a colonial economy but consolidated its sovereignty.[50] British Crown rule began in 1858. The rights promised to Indians were granted slowly,[51][52] but technological changes were introduced, and modern ideas of education and the public life took root.[53] A nationalist movement emerged in India, the first in the non-European British Empire and an influence on other nationalist movements.[54][55] Noted for nonviolent resistance after 1920,[56] it became the primary factor in ending British rule.[57] In 1947, the British Indian Empire was partitioned into two independent dominions,[58][59][60][61] a Hindu-majority dominion of India and a Muslim-majority dominion of Pakistan. A large-scale loss of life and an unprecedented migration accompanied the partition.[62]

India has been a federal republic since 1950, governed through a democratic parliamentary system. It is a pluralistic, multilingual and multi-ethnic society. India's population grew from 361 million in 1951 to over 1.4 billion in 2023.[63] During this time, its nominal per capita income increased from US$64 annually to US$2,601, and its literacy rate from 16.6% to 74%. A comparatively destitute country in 1951,[64] India has become a fast-growing major economy and a hub for information technology services, with an expanding middle class.[65] India has reduced its poverty rate, though at the cost of increasing economic inequality.[66] It is a nuclear-weapon state that ranks high in military expenditure. It has disputes over Kashmir with its neighbours, Pakistan and China, unresolved since the mid-20th century.[67] Among the socio-economic challenges India faces are gender inequality, child malnutrition,[68] and rising levels of air pollution.[69] India's land is megadiverse with four biodiversity hotspots.[70] India's wildlife, which has traditionally been viewed with tolerance in its culture,[71] is supported in protected habitats. """


# Small Chunk size
small_splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=2)
small_chunks = small_splitter.split_text(document)


# Large Chunk size
large_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
large_chunks = large_splitter.split_text(document)


embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Create Vector Stores 
small_vector_store=Chroma.from_texts(small_chunks, embedding_model)
large_vector_store=Chroma.from_texts(large_chunks, embedding_model)


query="what is India?"


small_results = small_vector_store.similarity_search(query, k=1)
large_results = large_vector_store.similarity_search(query, k=1)


print("smaller Chunks")
print(small_chunks)

print("Larger Chunks")
print(large_chunks)

print("---------------------------------------------------")
print("Results")
print("---------------------------------------------------")

print("smaller Chunks:", small_results)
print("Larger Chunks:", large_results)
