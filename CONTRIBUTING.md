## Contributing

Shrabanti Bala Joya, Sarisha Das, Omowunmi Obadero, Mantram Sharma

We welcome and appreciate all contributions. We make sure all contributions are cited and credited properly.

### Example Contributions

Ways you can contribute to this work:

- Report bugs
- Fixing issues
- Adding new features
- Improving documentation
- Sharing suggestions and feedback


### Developer Dependencies
conda (>= 24.11.0)  
conda-lock (>= 2.5.7)  


### Pull Request Guidelines
- Open a PR only after your changes are complete, tested, and documented where necessary.
- Keep your PR focused on a single update or fix, and include a clear description of what you changed.
- Ensure all checks and tests pass before requesting a review.
- New code should follow the PEP8 [style guide](https://peps.python.org/pep-0008/)


### Instructions for Installing Conda and Conda-Lock

Use Miniforge platform to install Conda. Download Miniforge for your operating system from https://conda-forge.org/download/. Install Miniforge by running the downloaded .exe file in Windows or run the following command in your terminal locally. Adjust the installer name as required.

```
bash ${HOME}/Downloads/Miniforge3.sh -b -p "${HOME}/miniforge3"
```

After installation run the following command to initialize conda

```
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate
conda init
```

Restart your terminal.

Once conda is installed, run the command below to install conda lock

```
conda install -c conda-forge conda-lock
```

### Instructions for Adding New Dependencies  

Open in your terminal, navigate to the root directory. To ensure the base is active run:
```
conda activate base
```

Create a conda environment called "YOURENV" using the "conda-lock.yml" by running in your terminal:

```
conda-lock install -n YOURENV conda-lock.yml
```
#### Change the local environment

1. Run the command below in terminal to activate the environment.

```
conda activate YOURENV
```

2. To update a single package to the latest version compatible with the version constraints in the source, run in your terminal:

```
conda-lock lock  --lockfile conda-lock.yml --update PACKAGE
```

3. To re-solve the entire environment, e.g. after changing a version constraint in the source file, run:

```
conda-lock -f environment.yml --lockfile conda-lock.yml
```

4. At root directory, update environment.yml using:

```
conda env export --from-history > environment.yml
```

> Note: Manually make sure to add version numbers from the environment to the created yml file

5. Use Conda-lock to solve and lock the updated environment. We are using Linux-64 because that's the operating system of our docker image:

```
    conda-lock lock --file environment.yml
    conda-lock -k explicit --file environment.yml -p linux-64
```
#### Update Docker Build

6. Re-build the docker image from the project root directory using a meaningful name (replace {YOUR-IMAGE-NAME} accordingly) and use the updated container locally. If you believe this new dependency should be included in the project repository, please open a pull request and we'll update the Docker image on Docker Hub

```
    docker build --tag {YOUR-IMAGE-NAME} .
```

7. Edit the `docker-compose.yml` file (living at root of directory), replace the image name with {YOUR-IMAGE-NAME}.
    
> For example, replace `"image: risha-daz/heart-disease-predictor:55f7470"` with `"image: {YOUR-IMAGE-NAME}"`

8. Running the docker image you just built at root directory

```
    docker compose up
 ```
## Report Bugs

Report bugs at <https://github.com/sjbalagit/Heart-Disease-Predictor---Group16/issues>.

## Fix Bugs

Look through the GitHub issues for bugs. Anything labelled with `bug` and
`help wanted` is open to whoever wants to implement it. When you decide to work on such
an issue, please assign yourself to it and add a comment that you'll be working on that,
too. If you see another issue without the `help wanted` label, just post a comment, the
maintainers are usually happy for any support that they can get.

## Implement Features

Look through the GitHub issues for features. Anything labelled with
`enhancement` and `help wanted` is open to whoever wants to implement it. As
for [fixing bugs](#fix-bugs), please assign yourself to the issue and add a comment that
you'll be working on that, too. If another enhancement catches your fancy, but it
doesn't have the `help wanted` label, just post a comment, the maintainers are usually
happy for any support that they can get.

## Write Documentation

Just [open an issue](<https://github.com/sjbalagit/Heart-Disease-Predictor---Group16/issues>) to let us know what you will be working on.

## Submit Feedback

The best way to send feedback is to file an issue at <https://github.com/sjbalagit/Heart-Disease-Predictor---Group16/issues>. If your feedback fits the format of one of
the issue templates, please use that. 

## Code of Conduct
- Treat all contributors with respect and professionalism.  
- Communicate clearly, politely and constructively.  
- Report issues, bugs or concerns in a constructive manner.

## Attributions

This contributing guide was adapted from the repository _[AI Bias in Farming](https://github.com/skysheng7/AI_bias_in_farming/blob/main/CONTRIBUTING.md)_ by Sky Sheng.
