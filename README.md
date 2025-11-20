# Lecture Summarization using fine-tune BART
Abstract:
This proposal addresses the challenge of generating concise, accurate, and domain-specific abstractive summaries for lectures using advanced natural language processing (NLP) techniques. While large language models (LLMs) like BART are highly effective for general summarization, they often struggle with specialized tasks such as lecture summarization, which requires handling unique terminology, theoretical concepts, and teaching methods that need tailored adaptation. Our approach combines parameter-efficient LoRA fine-tuning with concepts from Generative Adversarial Networks (GANs) to improve the domain specificity and coherence of the generated summaries. By leveraging lecture transcripts from various fields, we aim to bridge the gap between general-purpose LLMs and the specialized requirements of educational lecture summaries.
## Table of Contents
- [colab_example](#colab)
- [Installation](#installation)
- [run](#run)
- [License](#license)

## Colab
see the [Example_notebook.ipynb](https://colab.research.google.com/drive/1FK3H00CeCTO054lAmXz99yMBhcNj_1fQ?usp=sharing) *random seed has changed after the colab notesbook is upload.
## Installation
1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/project-name.git
    ```
2. **Navigate into the project directory**:
    ```bash
    cd "./COMP_8730_Project"
    ```
3. **Install dependencies**:
    ```bash
    pip install -r requirement.txt
    ```
   
## run
1. **train the model** from scratch with batch_size=5, num_epoch=1, and save both Generator and Discriminator model to ./SaveModel and save generator every epoch to ./Testmodel
    ```bash
    python main.py -o train -b 5 -e 1 -save true -l false -mode GAN
    ```
2. **continue train the model** from Generator_Path and Discriminator_Path with batch_size=5, num_epoch=1, and save both Generator and Discriminator model to ./SaveModel with continue_trained as prefix
    ```bash
    python main.py -o train -b 5 -e 1 -save true -l true -g 'Generator_Path' -d 'Discriminator_Path'
    ```
3. **predict** using the model from Generator_Path with input_text file from Input_Path and save a txt doc in ./Summary with summary of as prefix
    ```bash
    python main.py -o predict -g 'Generator_Path' -i 'Input_Path
    ```
4. **train the lora model** from scratch with batch_size=5, num_epoch=1, and save Generator every epoch to ./BARTmodel
    ```bash
    python main.py -o train -b 5 -e 1 -save true -l false -mode BART
    ```
5. **Evaluation** using the model from Generator ckpt calculate the ROUGE score of the model using test datasets. If -base true it will evaluate the base model.
    ```bash
    python main.py -o evaluate -g 'Generator_Path'
    ```
6. **Data augmentation** using the Lecture_aug.py to create augmentation datasets from Lecture datasets, num indicate how many data you want augmented.
    ```bash
    from Lecture_aug import data_aug
    data_aug(900)
    ```
## ckpt
1. Here is the google drive link to ckpt BART is fine-tune with Lora BART and BARTGAN is training using GAN
    [ckpt](https://drive.google.com/drive/folders/1qekUQg5TTpDGvjr1wOI25rxmFJ_7_Ta1?usp=sharing)
## License
1. This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For further inquiries, please contact [Zhang Shiqi](ZHANG3T3@uwindsor.ca).
