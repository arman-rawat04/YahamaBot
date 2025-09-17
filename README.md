# YamahaBot

## Installation

Install the following versions:

- Node JS Version: v20.9.0
- Python Version: 3.12

Navigate to `client/` and run:


npm install


This will install all the npm modules listed in `package.json`.

Create a virtual environment and install all the libraries mentioned in `requirements.txt` by running:

`pip install -r requirements.txt`


## How to Run

In the current directory, in the virtual environment, open the terminal and enter the following command to open the backend:

`uvicorn test:app --port 6969 --host 0.0.0.0`

Navigate to `client/` and run the following in the virtual environment to run the frontend:

`npm run dev`


Now visit [http://localhost:3000/](http://localhost:3000/) to use the DocBot.

