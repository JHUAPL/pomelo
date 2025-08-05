```
Current Dev Branch: new_classifer
```
# Basic Things

1. This is a repo with multiple people working on it. Please develop on your own branch and then merge into the current dev branch
2. Please test your code before committing. This is how bugs propogate. 
3. Please format your code with black. 

# Workflow
1. Assign yourself an issue
    - please do this 
    - I dont want to overlap on issues and commit on the same things
2. Make a branch for that issue
    - if you develop on the current dev branch, I will find you 
3. Work on that branch
    - To create a new feature the process of adding it to the workflow is simple
        - Create a config for it using the `config/base_config.py` as a base class
        - Add the config name to `config/pipelinesetup.py`
        - Add config class to `config/pomelo.py`
        - Add function to `drivers/pomelodriver.py`
    - If its a smaller change than use your best judgement
4. Add tests if need be
5. Test your code once finished
6. Ask for a merge req
    - DO NOT I REPEAT DO NOT MERGE WITHOUT A SECOND SET OF EYES
7. After the approval process merge it right in!
8. Repeat

# Testing and Tests
1. This is very important. Please test your code that you changed before adding it to the repo
2. If you add something with deterministic outputs, write a regular unit test 
3. If you add something with more random outputs, check that all the steps complete with a test (check if not None or the length)
