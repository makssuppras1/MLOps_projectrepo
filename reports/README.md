# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

`![my_image](figures/<image>.<extension>)`

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

or

```bash
uv add typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to either fill out the `requirements.txt`/`requirements_dev.txt` files or keeping your
    `pyproject.toml`/`uv.lock` up-to-date with whatever dependencies that you are using (M2+M6)
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [ ] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [ ] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [x] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12)
* [x] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [x] Add a linting step to your continuous integration (M17)
* [x] Add pre-commit hooks to your version control setup (M18)
* [x] Add a continues workflow that triggers when data changes (M19)
* [x] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [x] Create a trigger workflow for automatically building your docker images (M21)
* [x] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [ ] Create a FastAPI application that can do inference using your model (M22)
* [ ] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [ ] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [ ] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Setup collection of input-output data from your deployed application (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [ ] Make sure all group members have an understanding about all parts of the project
* [ ] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

MLOps 130

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s204634, s204614, s204598

### Question 3
> **Did you end up using any open-source frameworks/packages not covered in the course during your project? If so**
> **which did you use and how did they help you complete the project?**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We used the **Transformers** library from Hugging Face, specifically **DistilBert**, as our base model for text classification tasks on scientific papers. This library provided pre-trained transformer models and tokenizers that significantly accelerated our NLP pipeline development.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We used **UV** for managing our dependencies. Our dependencies are defined in the `pyproject.toml` file for main dependencies and dependency-groups for development dependencies. The exact versions are locked in the `uv.lock` file for reproducible builds. To get a complete copy of our development environment, a new team member would need to: 
1) Install UV package manager (following the [official guide](https://docs.astral.sh/uv/getting-started/installation/)), 
2) Clone the repository, 
3) Run `uv sync` to install all dependencies exactly as specified in the lock file, 
4) Download the data to your local repository buy running the following command in the terminal: ``uv run sh curl_arxiv-scientific-research-papers-dataset``
5) Optionally run `uv sync --group dev` to include development dependencies like pytest, coverage, and pre-commit.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

From the cookiecutter template [mlops_template](https://github.com/SkafteNicki/mlops_template) we filled out the **src/pname/** folder with core modules including `data.py` for dataset handling, `model.py` for our DistilBert-based model, `train.py` for training procedures, `api.py` for FastAPI implementation, `evaluate.py` for model evaluation, `metrics.py` for performance metrics, `visualize.py` for plotting, and `profiler.py` for performance profiling. The **configs/** folder contains Hydra configuration files: `config.yaml`, `model_conf.yaml`, `training_conf.yaml`, `sweep.yaml`, and experiment-specific configs in the `experiment/` subfolder. We implemented three dockerfiles in **dockerfiles/**: `train.dockerfile`, `evaluate.dockerfile`, and `api.dockerfile`. The **tests/** folder contains unit tests: `test_data.py`, `test_model.py`, `test_training.py`, and `test_api.py`. We kept the **docs/** folder with MkDocs setup and the **notebooks/** folder for analysis. 

We deviated from the template by adding several project-specific files: `tasks.py` for invoke commands, guide files (`LOGGING_GUIDE.md`, `profiling_guide.md`, `config_guide.md`), a data download script (`curl_arxiv-scientific-research-papers-dataset`), and various output directories. The core template structure was maintained while adding these practical extensions for our specific MLOps workflow.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

--- question 6 fill here ---

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:
I total we have made 17 tests (5 in test_data, 8 in test_model, 4 in test_training) with the focus on testing the code for our data, model and train scripts.

*test_data.py* focuses on the data processing/preprocessing pipeline; Validating that the ArXivDataset classes are initialized correctly, that the expected output files are created, that the train/val/tests split ratios are correct, and checks that the category-to-label mapping are created correctly for classification.

*test_model.py* focuses on model architecture and behavior; Initialization, forward passes, output validation, encoder freezing, parameter counting, and gradient flow.

*test_training.py* focuses on training pipeline and reproducibility; Seed reproducibility, batch collation, and training dynamics.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The total code coverage of our code is **48%**, covering 136 out of 285 total statements across our main modules. We are far from 100% coverage, and even if we achieved 100%, we would not trust it to be completely error-free. Code coverage is a quantitative metric showing which lines were executed during testing, but doesn't guarantee test quality or correctness. High coverage can provide false confidence if tests only verify code execution.

Our current coverage breakdown shows: `data.py` at 61%, `model.py` at 71%, and `train.py` at 27%. This indicates good coverage of our core data processing and model functionality, but limited coverage of the training pipeline. Coverage doesn't account for edge cases, integration issues, or logical errors in test assertions themselves. Modules like API, evaluation, and visualization remain completely untested, representing areas for test suite expansion.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We made use of both branches and pull requests (PRs) in our project. Rather than having individual branches per group member, we implemented **feature-based branches** where each branch corresponded to a specific feature or functionality being developed. This approach was chosen because we implemented co-coding practices where multiple team members could work together on the same features. We used pull requests to merge these feature branches back into the main branch, which allowed us to keep the main branch as up-to-date as possible while maintaining smaller, focused changes per PR.
Additionally, we set up our GitHub repository to require a **minimum of 2 group members to approve a PR** before it could be merged, following the methods taught in module 17 of the course material. This workflow helped us maintain better version control by ensuring each PR contained a cohesive set of changes related to a specific feature, making code reviews more manageable and reducing conflicts during merges.


### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
DVC was used for managing the data in our project. We configured DVC with Google Clound Storage (GCS) as our remote storage backend, which allowed us to version control our dataset effectively. For collaboration between team members it offers the advantage of ensuring that everone is working on the same data, for example instead of downloading the dataset and running the preprocessing every time, each team member simply gets the dataset from the cloud. 

In truth it might have been overkill to do data version control for this specific dataset as it is static, and the preprocessing was relatively banal and unlikely to change during the project. If our dataset consisted every scientific paper and was updated every time a new paper was published it would definately make data version control a requirement for this project.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

TODO describe in detail some of the important tests that we are running.

The Continuous integration setup consists of 'GitHub Actions' workflows located in the tests.yaml file. The file runs when a pull requests or a push to main is made. It runs three jobs, unit testing, linting, and basic packaging checks.

* **Testing**: checks out the repo, sets up uv, caches dependencies, installs project dependencies, installs a specified PyTorch version for testing, runs pytest, and then runs coverage. The test across multiple platforms (Ubuntu, MacOS and Windows) and python versions (3.12, 3.13) to ensure a minimum level of compatibility.

* **Linting and formatting**: Handled in the tests.yaml file as 'jobs'. Ubuntu sets up uv and runs Ruff linting and formatting check.

* **Build**: builds Docker images; the job is defined but details are in the same file.

Github's Caching is utilized to store the python packages from the uv.lock file, so that subsequent runs can use the stored packages and skip the process of installing them again. This means that the first run will be relatively lengthy (a couple of minutes) as it has to install all the packages, but subsequent tests can be performed in seconds. The time saved by caching grows exponentially as the number of tests increase. 

An example of a triggered for our workflow can be seen [here](https://github.com/makssuppras1/MLOps_projectrepo/actions/runs/21134839770).

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We used **Hydra** for configuration management with hierarchical YAML config files. The main `config.yaml`(configs/config.yaml) loads default configurations for model, training, and experiments. Specific experiments are defined in *configs/experiment/* folder (`fast.yaml`, `balanced.yaml`, `optimized_distilbert.yaml`, etc.). Each experiment overrides base training parameters like batch_size, epochs, max_samples, and learning rates.

**Example**: To run the fast experiment configuration:
```bash
uv run src/pname/train.py experiment=fast
```

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

--- question 13 fill here ---

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

--- question 15 fill here ---

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

--- question 16 fill here ---

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

***Google Cloud Storage (GCS)***: Object storage service used to store raw data as well as tranining data, serve as DVC's remote storage for version-controlled datasets, and stage source code for Cloud Build operations.

***Compute Engine***: Virtual machine service used to create and manage VM instances for running the ML training. The VM is placed in ``europe-west1-d`` to minimize the distrance and therby secure a lower cost. Furthermore, the machine typs is set to ``e2-medium``. 

***Artifact Registry***: Container registry service used to store and version Docker images, enabling image distribution and deployment across the project.

***Cloud Build***: CI/CD service used to build Docker images in the cloud from source code, automatically handling the build process and pushing images to Artifact Registry without requiring local Docker installation.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

For this project, we used Google Compute Engine (GCE) to move our computations from a local environment to the cloud. To run the training of our mode, we deployed an ``n1-standard-4`` instance (4 vCPUs, 15 GB memory) in the ``europe-west1-b`` zone.

To manage our data, we linked the VM to Google Cloud Storage (GCS) using DVC. We configured the VM's service account to securely pull versioned datasets from our bucket (``gs://mlops_project_data_bucket1``) without manual authentication. By using the version_aware setting in our DVC config, we ensured that our data remains organized and accessible within the GCP ecosystem. This setup allows us to treat the VM as a reproducible environment where we can clone our code, run dvc pull to fetch the exact data version needed, and execute training scripts in a scalable cloud infrastructure.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

![bucket_20012026](figures/bucket_20012026.png)

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

![registry_20012026](figures/registry_20012026.png)

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

![build_20012026](figures/build_20012026.png)

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

We managed to train our model in the cloud using **Vertex AI**. We chose to migrate to Vertex AI from Compute Engine because it provides managed machine learning infrastructure with better integration for ML workloads.

Our training setup works as follows: We created two Vertex AI training configurations - `vertex_ai_config_cpu.yaml` for CPU-only training using *n1-highmem-4* machines, and `vertex_ai_config_gpu.yaml` for GPU-accelerated training using *n1-standard-4* machines. Both configurations specify our custom Docker container from Artifact Registry (`europe-west1-docker.pkg.dev/dtumlops-484310/container-registry/train:latest`).

The training process is initiated through `vertex_ai_train.yaml` which submits custom training jobs to Vertex AI. Our Cloud Storage bucket (`mlops_project_data_bucket1`) is automatically mounted at `/gcs/mlops_project_data_bucket1/` during training, providing direct access to our DVC-managed datasets. We use Secret Manager to securely inject the WANDB_API_KEY for experiment tracking.

This setup allows us to run scalable training jobs in the europe-west1 region without managing underlying infrastructure, while maintaining full integration with our data versioning, containerization, and experiment tracking pipeline.

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

--- question 23 fill here ---

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

--- question 24 fill here ---

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- question 26 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

--- question 27 fill here ---

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

--- question 28 fill here ---

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 29 fill here ---

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- question 30 fill here ---

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

--- question 31 fill here ---
