# Topic View
This is an app which allows a user to upload a csv dataset, generate a topic model, and visually explore the results.

### End Users:
* **Data Scientists** - to perform initial topic exploration/ideation in preparation for detailed topic modelling. 
* **Business Analystists** - to better understand topic clustering for segmenting customer behavior.
* **Business Stakholders** - to begin understanding the main subjects/topics, and words customers are using.



## Project Overview

```
├── build                   # Compiled app bundle
├── data                    # Static files used primarily for local development
├── docs                    # Documentation 
├── sql_scripts             # SQL used to query real-time information
├── src                     # Application source code 
├── static                  # Non-changing assets
├── venv                    # Virtual python environment
├── .appignore              # Files to not publish to the AI Cloud
├── .gitignore              # Files to not check into GitHub
├── app.toml                # AI Cloud configuration
├── Makefile                # For local code compilation 
├── README.md               
└── requirements.txt
```

## Contributing
When working on this application, please complete the following items:

- [ ] Testing
    - [ ] Ensure that the app can run successfully locally
    - [ ] Ensure that the app can run in the AI Cloud using `h2o bundle deploy`

- [ ] Clean up
    - [ ] Add folders and files to the `.gitignore` which should not be checked into GitHub
    - [ ] Add folders and files to the `.appignore` which should do not need to be added to the AI Cloud
    - [ ] Run `make format` to ensure the codebase stays consistent using black and isort

- [ ] Documentation
    - [ ] Document any new App Secrets in the `README.md`
    - [ ] Add any new developer information to the `README.md`
    - [ ] Read the `README.md`, is everything still relevant and true?
    - [ ] Add any new user features to the `docs/description_for_ai_cloud.md`
    - [ ] Read the `docs/description_for_ai_cloud.md`, is everything still relevant and true?
    - [ ] Replace any dated images in the `static` folder
    
- [ ] Release
    - [ ] Increment the app version number in the `app.toml`
    - [ ] Create a new bundle using `h2o bundle` and this file into the `build` folder, no need to delete old bundles
    - [ ] Deploy the app to the AI Cloud

- [ ] Open a pull request (PR) in GitHub
    - [ ] Assign a reviewer
    - [ ] Assign at least one `area` label and `type` label
    - [ ] Use a descriptive title 
    - [ ] Include a list of what your PR does in the description
    - [ ] Include `RELNOTES=brief explaination` in the description
    
    
## Local Development
Setup your local python environment. Note that python 3.7 is specified as this is the default for the H2O AI 
Hybrid Cloud.

```shell script
make setup
```

Run your app for local development **after setting up the key information in Make File.**
```shell script
make run
```

## Run in the AI Cloud

### App Secrets
This application must be parameterized with the telemetry credentials, steam admin credentials, and okta credentials
for the Trial Cloud to run in any H2O AI Cloud.

```sh
h2o secret create cloud-telemetry-db-credentials --from-literal=url="" --from-literal=username="" --from-literal=password="" --from-literal=database=""

h2o secret create cloud-steam-admin --from-literal=url="" --from-literal=token=""

h2o secret create cloud-okta --from-literal=url="" --from-literal=token="" --from-literal=client-id=""
```