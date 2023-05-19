[![Build Status](https://travis-ci.org/joommf/conda-install.svg?branch=master)](https://travis-ci.org/joommf/conda-install)

# STEP 1
## conda install

Jupyter OOMMF can be installed via conda. The basic steps are:

- install Anaconda (use "miniconda" if you are short of disk space). See for example https://docs.anaconda.com/anaconda/install/ for details

- run the following command to install Jupyter OOMMF

<pre>
    conda install -c conda-forge oommfc
</pre>


- you will also want to install the Jupyter Notebooks
<pre>
    conda install notebook
</pre>

  and maybe other python packages you want to use.


We are working on a conda package with name 'joommf' which will
install the notebook automatically.

## Problems?

### Unthreaded tk version

Symptom

<pre>
        if val.returncode is not 0:
>           raise RuntimeError("Some problem calling OOMMF.")
E           RuntimeError: Some problem calling OOMMF.
/opt/conda/lib/python3.6/site-packages/oommfc/oommf.py:58: RuntimeError
----------------------------- Captured stdout call -----------------------------
2017/11/7 11:32: Calling OOMMF (stdprobfmr/stdprobfmr.mif) ... [0.4s]
Error when executing:
	command: /opt/conda/bin/oommf boxsi +fg stdprobfmr/stdprobfmr.mif -exitondone 1
	stdout: b'Boxsi run end.\n'
	stderr: b'<364> oommf.tcl 1.2.1.0  panic:\nchild process exited abnormally\n'
</pre>

If you see output like the above, it is possible that `conda` uses its own package `tk` (from channel `default`) rather than from channel `conda-forge`. The tk on the `conda-forge` channel supports threading, which is needed for OOMMF to work. The following command should solve the problem:
<pre>
conda update -c conda-forge tk
</pre>

#### More details on diagnosing the situation for the curious minds
If you want to diagnose which tk package is installed, you can use `conda search -c conda-forge tk` and search for the lines starting with `tk`:
<pre>
tk                           8.5.13                        0  defaults
                             8.5.13                        1  defaults
                             8.5.15                        0  defaults
                             8.5.18                        0  defaults
                             8.5.19                        0  conda-forge
                             8.5.19                        1  conda-forge
                          *  8.5.19                        2  conda-forge
                             8.6.6                         0  conda-forge
                             8.6.6                         1  conda-forge
                             8.6.6                         2  conda-forge
                             8.6.6                         3  conda-forge
                             8.6.6                         4  conda-forge
                             8.6.6                         5  conda-forge
</pre>
The installed version is the one with the `*`. In the above example, this is the right package from the `conda-forge` channel (right most column).

We have tracked down this problem [here](https://github.com/joommf/joommf/issues/81#issuecomment-342483535), but haven't found a good solution yet (other than issueing the above `conda update` command manually).

## Which version of OOMMF is provided?

On Linux and OS X, the conda-forge install (as described above) provides oommfc compile from the sources available in http://github.com/fangohr/oommf/ . These include the bulkDMI OOMMF extension.

On Windows, the conda-forge install (as describe above) provides the OOMMF as is distributed from NIST (see https://github.com/conda-forge/oommf-feedstock/blob/master/recipe/meta.yaml#L10).



# STEP 2

## Managing conda
Verify that conda is installed and running on your system by typing:

		conda --version
Conda displays the number of the version that you have installed. You do not need to navigate to the Anaconda directory.

EXAMPLE: conda 4.7.12

### Note

If you get an error message, make sure you closed and re-opened the terminal window after installing, or do it now. Then verify that you are logged into the same user account that you used to install Anaconda or Miniconda.

Update conda to the current version. Type the following:

		conda update conda
Conda compares versions and then displays what is available to install.

If a newer version of conda is available, type y to update:

		Proceed ([y]/n)? y
#### Tip

We recommend that you always keep conda updated to the latest version.

## Managing environments
Conda allows you to create separate environments containing files, packages, and their dependencies that will not interact with other environments.

When you begin using conda, you already have a default environment named base. You don't want to put programs into your base environment, though. Create separate environments to keep your programs isolated from each other.

1. Create a new environment and install a package in it.

We will name the environment snowflakes and install the package BioPython. At the Anaconda Prompt or in your terminal window, type the following:

		conda create --name snowflakes biopython
Conda checks to see what additional packages ("dependencies") BioPython will need, and asks if you want to proceed:

		Proceed ([y]/n)? y
Type "y" and press Enter to proceed.

2. To use, or "activate" the new environment, type the following:

Windows: conda activate snowflakes

macOS and Linux: conda activate snowflakes

#### Note

conda activate only works on conda 4.6 and later versions.

For conda versions prior to 4.6, type:

Windows: activate snowflakes

macOS and Linux: source activate snowflakes

Now that you are in your snowflakes environment, any conda commands you type will go to that environment until you deactivate it.

3. To see a list of all your environments, type:

		conda info --envs
		
A list of environments appears, similar to the following:

	conda environments:

    base           /home/username/Anaconda3
    snowflakes   * /home/username/Anaconda3/envs/snowflakes
#### Tip

The active environment is the one with an asterisk (*).

4. Change your current environment back to the default (base): conda activate

#### Note

For versions prior to conda 4.6, use:

Windows: activate

macOS, Linux: source activate

#### Tip

When the environment is deactivated, its name is no longer shown in your prompt, and the asterisk (*) returns to base. To verify, you can repeat the conda info --envs command.

# Managing Python
When you create a new environment, conda installs the same Python version you used when you downloaded and installed Anaconda. If you want to use a different version of Python, for example Python 3.6, simply create a new environment and specify the version of Python that you want.

1. Create a new environment named "snakes" that contains Python 3.6:

		conda create --name snakes python=3.6
		
When conda asks if you want to proceed, type "y" and press Enter.

2. Activate the new environment:

Windows: conda activate snakes

macOS and Linux: conda activate snakes

#### Note

conda activate only works on conda 4.6 and later versions.

For conda versions prior to 4.6, type:

Windows: activate snakes

macOS and Linux: source activate snakes

3. Verify that the snakes environment has been added and is active:

		conda info --envs
		
Conda displays the list of all environments with an asterisk (*) after the name of the active environment:

## conda environments:
#
base                     /home/username/anaconda3
snakes                *  /home/username/anaconda3/envs/snakes
snowflakes               /home/username/anaconda3/envs/snowflakes
The active environment is also displayed in front of your prompt in (parentheses) or [brackets] like this:

		(snakes) $
4. Verify which version of Python is in your current environment:

		python --version
5. Deactivate the snakes environment and return to base environment: conda activate

Note

For versions prior to conda 4.6, use:

Windows: activate

macOS, Linux: source activate



# STEP 3: 

## Install some libraries

		$ pip install -r requirements.txt


# STEP 4: 
## Run Python Project in Command Line

### Using the python Command
To run Python scripts with the python command, you need to open a command-line and type in the word python, or python3 if you have both versions, followed by the path to your script, just like this:

		$ python model_new_pro.py
    
If everything works okay, after you press Enter, you’ll see the output on your screen. That’s it! You’ve just run your first Python script!

If this doesn’t work right, maybe you’ll need to check your system PATH, your Python installation, the way you created the hello.py script, the place where you saved it, and so on.

This is the most basic and practical way to run Python scripts.

The code above is to train a model, if you want to test the model, run the code below:

                $ python data_testing.py
 
 #### Note: The training and testing data is used with these columns, including: ['site', 'car_name', 'brand', 'grade', 'model', 'color', 'year_car', 'engine', 'odo', 'score', 'start_price', 'auction_site'], so you should change these attributes in <code>data_testing.py</code> and <code>model_new_pro.py</code>


### Redirecting the Output
Sometimes it’s useful to save the output of a script for later analysis. Here’s how you can do that:

		$ python model_new_pro.py > output.txt or python data_testing.py > output.txt
    
This operation redirects the output of your script to output.txt, rather than to the standard system output (stdout). The process is commonly known as stream redirection and is available on both Windows and Unix-like systems.

If output.txt doesn’t exist, then it’s automatically created. On the other hand, if the file already exists, then its contents will be replaced with the new output.

Finally, if you want to add the output of consecutive executions to the end of output.txt, then you must use two angle brackets (>>) instead of one, just like this:

		$ python model_new_pro.py >> output.txt or python data_testing.py >> output.txt
		
Now, the output will be appended to the end of output.txt.

### Stop python file
To exit interactive mode, you can use one of the following options:

	- quit() or exit(), which are built-in functions
	- The Ctrl+Z and Enter key combination on Windows, or just Ctrl+D on Unix-like systems
Note: The first rule of thumb to remember when using Python is that if you’re in doubt about what a piece of Python code does, then launch an interactive session and try it out to see what happens.

If you’ve never worked with the command-line or terminal, then you can try this:

	On Windows, the command-line is usually known as command prompt or MS-DOS console, and it is a program called cmd.exe. The path to this program can vary significantly from one system version to another.

	A quick way to get access to it is by pressing the Win+R key combination, which will take you to the Run dialog. Once you’re there, type in cmd and press Enter.

	On GNU/Linux (and other Unixes), there are several applications that give you access to the system command-line. Some of the most popular are xterm, Gnome Terminal, and Konsole. These are tools that run a shell or terminal like Bash, ksh, csh, and so on.

	In this case, the path to these applications is much more varied and depends on the distribution and even on the desktop environment you use. So, you’ll need to read your system documentation.

	On Mac OS X, you can access the system terminal from Applications → Utilities → Terminal.

## Run Python Scripts From an IDE or a Text Editor
When developing larger and more complex applications, it is recommended that you use an integrated development environment (IDE) or an advanced text editor.

Most of these programs offer the possibility of running your scripts from inside the environment itself. It is common for them to include a Run or Build command, which is usually available from the tool bar or from the main menu.

Python’s standard distribution includes IDLE as the default IDE, and you can use it to write, debug, modify, and run your modules and scripts.

Other IDEs such as Eclipse-PyDev, PyCharm, Eric, and NetBeans also allow you to run Python scripts from inside the environment.

Advanced text editors like Sublime Text and Visual Studio Code also allow you to run your scripts.

To grasp the details of how to run Python scripts from your preferred IDE or editor, you can take a look at its documentation.

## Run Python Scripts From a File Manager
Running a script by double-clicking on its icon in a file manager is another possible way to run your Python scripts. This option may not be widely used in the development stage, but it may be used when you release your code for production.

In order to be able to run your scripts with a double-click, you must satisfy some conditions that will depend on your operating system.

Windows, for example, associates the extensions .py and .pyw with the programs python.exe and pythonw.exe respectively. This allows you to run your scripts by double-clicking on them.

When you have a script with a command-line interface, it is likely that you only see the flash of a black window on your screen. To avoid this annoying situation, you can add a statement like input('Press Enter to Continue...') at the end of the script. This way, the program will stop until you press Enter.

This trick has its drawbacks, though. For example, if your script has any error, the execution will be aborted before reaching the input() statement, and you still won’t be able to see the result.

On Unix-like systems, you’ll probably be able to run your scripts by double-clicking on them in your file manager. To achieve this, your script must have execution permissions, and you’ll need to use the shebang trick you’ve already seen. Likewise, you may not see any results on screen when it comes to command-line interface scripts.

Because the execution of scripts through double-click has several limitations and depends on many factors (such as the operating system, the file manager, execution permissions, file associations), it is recommended that you see it as a viable option for scripts already debugged and ready to go into production.
