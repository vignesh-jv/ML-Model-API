# Sarcasm Classifier Deployed as a REST API using Flask

* [Flask Restful Documentation]()
___

## Procedure
1. Start a virtual environment and install requirements
2. Run `build_model.py` to build the sarcasm classifier.
3. Run `app.py` which is the API application that will be deployed
4. Update requirements.txt as you write the code


## File Structure
* app_name
  * app.py: Flask API application
  * model.py: class object for classifier
  * build_model.py: imports the class object from `model.py` and initiates a new model, trains the model, and pickle
  * util.py: helper functions for `model.py`
  * requirements.txt: list of packages that the app will import
  * lib
      * data: directory that contains the train and test files
      * models: directory that contains the pickled model files


## Deployin the API
1. Run the Flask API locally. Go to directory with `app.py`.

```bash
python app.py
```

## Appendix

### Virtual Environment
1. Create new virtual environment
```bash
cd ~/.virtualenvs
virtualenv -p python3 name-of-env
```
2. Activate virtual environment
```
source env/bin/activate
```
3. Go to app.py directory where `requirements.txt` is also located
4. Install required packages from `requirements.txt`
```bash
pip install -r requirements.txt
```
You will only have to install the `requirements.txt` when working with a new virtual environment.
### Links to setup and installations and rough drafts
https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
