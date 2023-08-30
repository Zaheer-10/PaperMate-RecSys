<center>
  <img src="https://github.com/Zaheer-10/PaperMate-RecSys/blob/main/PM_Resources/Images/PaperMate_animated.gif" alt="PaperMate-logo" align="middle">
</center>

## What is PaperMate?

PaperMate is a cutting-edge NLP project that transforms the way you find, read, and enjoy research papers in machine learning, natural language processing, computer vision, and artificial intelligence. With PaperMate, you can simply tell or type what topics you are interested in, and get a curated list of arXiv papers organized by categories. PaperMate also uses smart algorithms to suggest papers that match your preferences. You can access the full papers, get summaries, visualize papers graphically, and have natural language chats about papers. PaperMate makes research more fun and easy to explore.

For example, suppose you are a researcher working on using transformers for computer vision tasks. You can enter some keywords or a short description of your interests into the recommendation system. The system will then analyze your input, compare it with the existing research, and recommend a set of papers that are most relevant to your research domain.

## Problem framing

Imagine how many research papers are written and shared by scientists in conferences and journals. It’s a huge amount! But for learners, researchers, and curious people, finding the papers that match their interests is like looking for a needle in a haystack. That’s why PaperMate is here to help.

PaperMate is like a clever friend that knows a lot about research. It makes things easier by recommending machine learning papers that suit your preferences. If you’re into machine learning, deep learning, or NLP, PaperMate is like having a pal who is an expert in those fields.

The main problem is that there’s so much research out there that it’s hard to stay updated. The old ways of finding papers are time-consuming. But PaperMate changes that. It uses smart technology to make research easy to find and fun to explore. It’s like having a super-intelligent friend who takes you on a journey through the world of science. 

<div align="center">
    <h2>YEARLY PAPER PUBLICATION ON ARXIV</h2>
    <img src="https://github.com/Zaheer-10/PaperMate-RecSys/blob/main/PM_Resources/Images/monthly_submission.png" alt="Paper 1" width="400">
    <img src="https://github.com/Zaheer-10/PaperMate-RecSys/blob/main/PM_Resources/Images/count.png" alt="Paper 2" width="400">
</div>

### Identifying the Machine Learning Problem

Our main challenge is to recommend research papers that suit a user’s interests. It’s like a puzzle: finding papers that match what the user wants to learn. This involves using natural language processing (NLP) to compare the content of papers and user preferences.

But we go beyond recommendations. With NLP and machine learning, we enable question answering – users can ask questions about a paper’s content, and the system answers. Summarization is also part of our solution, shrinking long papers into short summaries. NLP and ML work together to create a more interactive, efficient, and insightful research experience.

## Plan 

- `Data Collection`: arXiv papers data is scraped from the arXiv website.
- `Data Refinement`: Employ EDA, Data Processing, and Feature Engineering for optimal text data.
- `Embedding Process`: Utilize Hugging Face Sentence Transformer ([all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)) for meaningful text embeddings.
- `Similarity Assessment`: Calculate paper similarity via cosine similarity for both titles and abstracts.
- `Top Recommendations`: Identify 4 most similar papers for users, including abstracts and full-text links.
- `Continuous Enhancement`: Set up monthly pipeline to scrape new papers from arXiv and re-embed for updated recommendations.
- `Summarization`: Utilize ([MBZUAI/LaMini-Flan-T5-248M](https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M/tree/main)) for summarization, generating concise and informative summaries of extensive research papers.
- `Question Answering`: Leverage ([ggml-gpt4all-j-v1.3-groovy](https://huggingface.co/nomic-ai/gpt4all-j)) for question answering, enabling users to interact with papers through natural language queries and receive informative responses.
- `Online Access`: Deploy the application online for user access. (Deployment details will be added later.)

This holistic approach ensures PaperMate is a comprehensive and user-centric platform that enhances research exploration, understanding, and engagement. Watch for ongoing improvements as we continue to refine and optimize each component!

## Observations

### Distribution of titles's length

The average text length of a title is 73 characters, and its maximum length is 217 characters. Choosing a sentence-transformer model with a Max Sequence Length capability of over 217 characters would be ideal. The `all-MiniLM-L6-v2` model has a Max Sequence Length of 256, which is more than enough to process the entire title.

![Distribution-titles-length](https://github.com/Zaheer-10/PaperMate-RecSys/blob/main/PM_Resources/Images/titels_length.png)

## Architecture

PaperMate's architecture encompasses several essential components to create a seamless user experience:

- User Interface: Developed with Django's template system, HTML, CSS, JS, and Bootstrap, the user interface provides an interactive and visually appealing platform for users to engage with research papers.

- Backend Framework: Leveraging the Django framework, the backend server adeptly manages user requests, handles data processing, and orchestrates PaperMate's multifaceted functionalities.

- Recommendation Engine: At the core, a machine learning model facilitates content-based paper recommendations, tailoring suggestions to users' preferences and interests.

- Summarization and QA Models: Integration of the MBZUAI/LaMini-Flan-T5-248M summarization model and ggml-gpt4all-j-v1.3-groovy question answering model enriches the user journey with concise summaries and insightful paper interactions.

## Behind the Scenes: How PaperMate Operates

<table>
  <tr>
    <td align="center">
      <h2>Data Collection</h2>
      <img src="https://github.com/Zaheer-10/PaperMate-RecSys/blob/main/PaperMate_ui/GUI/static/images/data_collection_svg.excalidraw.svg" alt="Paper 1" width="400">
    </td>
    <td align="center">
      <h2>EDA & Data Pre-processing</h2>
      <img src="https://github.com/Zaheer-10/PaperMate-RecSys/blob/main/PaperMate_ui/GUI/static/images/eda_datapreprocessing.excalidraw.svg" alt="Paper 2" width="400">
    </td>
  </tr>
  <tr>
    <td align="center">
      <h2>Embedding Process</h2>
      <img src="https://github.com/Zaheer-10/PaperMate-RecSys/blob/main/PaperMate_ui/GUI/static/images/embeddings.svg" alt="Paper 3" width="400">
    </td>
    <td align="center">
      <h2>Paper Recommendation</h2>
      <img src="https://github.com/Zaheer-10/PaperMate-RecSys/blob/main/PaperMate_ui/GUI/static/images/recommendation.svg" alt="Paper 4" width="400">
    </td>
  </tr>
  <tr>
    <td align="center">
      <h2>Summarization</h2>
      <img src="https://github.com/Zaheer-10/PaperMate-RecSys/blob/main/PaperMate_ui/GUI/static/images/summarization.svg" alt="Paper 5" width="400">
    </td>
    <td align="center">
      <h2>Question Answering System</h2>
      <img src="https://github.com/Zaheer-10/PaperMate-RecSys/blob/main/PaperMate_ui/GUI/static/images/qa.svg" alt="Paper 6" width="400">
    </td>
  </tr>
</table>

## Report

The project report is available [here](). It provides a detailed explanation of the methodologies used, results obtained, and conclusions drawn from this project.

## Future Possibilities

Continual improvements might include:

- [ ] Adding more data sources.
- [ ] Adding new menu to select which sentence transformer to use.
- [ ] Building a Docker image for the app and deploy it on Docker Hub.
- [ ] Finishing writing the project report.

## Acknowledgments

I would like to extend my gratitude to arXiv for generously providing open access to their interoperability, which has significantly contributed to the success of this project.

## Conclusion

In conclusion, PaperMate is not just a project—it's a transformational tool that democratizes knowledge, redefines research, and empowers individuals to navigate the intricate corridors of academia with confidence and curiosity. The journey is ongoing, and as we continue to innovate and iterate, PaperMate remains dedicated to propelling us into a brighter and more insightful future.

Thank you for embarking on this journey with us.

Discover, Engage, Illuminate—Welcome to PaperMate.


<p align="center">
  Want to experience the magic of PaperMate? Visit our website!
  <br>
  <a href="Website_URL">
    <img src="https://github.com/Zaheer-10/PaperMate-RecSys/blob/main/PM_Resources/Images/PaperMate_animated.gif" alt="Visit Website" width="150">
  </a>
</p>

<hr>
