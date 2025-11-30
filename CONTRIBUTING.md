## Contributing

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

Open in your terminal, navigate to the root directory.

Create a conda environment called "YOURENV" using the "conda-lock.yml" by running in your terminal:

```
conda-lock install -n YOURENV conda-lock.yml
```
### Activate the conda environment

Run the command below in terminal to activate the environment.

```
conda activate YOURENV
```

To update a single package to the latest version compatible with the version constraints in the source, run in your terminal:

```
conda-lock lock  --lockfile conda-lock.yml --update PACKAGE
```

To re-solve the entire environment, e.g. after changing a version constraint in the source file, run:

```
conda-lock -f environment.yml --lockfile conda-lock.yml
```

At root directory, update environment.yml using:

```
conda env export --from-history > environment.yml
```

### Code of Conduct
- Treat all contributors with respect and professionalism.  
- Communicate clearly, politely and constructively.  
- Report issues, bugs or concerns in a constructive manner.   


## Attributions

This contributing guide was adapted from the repository _[AI Bias in Farming](https://github.com/skysheng7/AI_bias_in_farming/blob/main/CONTRIBUTING.md)_ by Sky Sheng.
