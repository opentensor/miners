#!/bin/bash

# Initialize variables
script="openvalidators/neuron.py"
autoRunLoc=$(readlink -f "$0")
proc_name="openvalidators_main_process" 
args=()
version_location="./openvalidators/__init__.py"
version="__version__"

old_args=$@

# Check if pm2 is installed
if ! command -v pm2 &> /dev/null
then
    echo "pm2 could not be found. To install see: https://pm2.keymetrics.io/docs/usage/quick-start/"
    exit 1
fi


# Loop through all command line arguments
while [[ $# -gt 0 ]]; do
  arg="$1"

  # Check if the argument starts with a hyphen (flag)
  if [[ "$arg" == -* ]]; then
    # Check if the argument has a value
    if [[ $# -gt 1 && "$2" != -* ]]; then
          if [[ "$arg" == "--script" ]]; then
            script="$2";
            shift 2
        else
            # Add '=' sign between flag and value
            args+=("'$arg'");
            args+=("'$2'");
            shift 2
        fi
    else
      # Add '=True' for flags with no value
      args+=("'$arg'");
      shift
    fi
  else
    # Argument is not a flag, add it as it is
    args+=("'$arg '");
    shift
  fi
done

# Check if script argument was provided
if [[ -z "$script" ]]; then
    echo "The --script argument is required."
    exit 1
fi

branch=$(git branch --show-current)            # get current branch.
echo watching branch: $branch
echo pm2 process name: $proc_name

# Check if script is already running with pm2
if pm2 status | grep -q $proc_name; then
    echo "The script is already running with pm2. Stopping and restarting..."
    pm2 delete $proc_name
fi

# Run the Python script with the arguments using pm2
echo "Running $script with the following pm2 config:"

# Join the arguments with commas using printf
joined_args=$(printf "%s," "${args[@]}")

# Remove the trailing comma
joined_args=${joined_args%,}

# Create the pm2 config file
echo "module.exports = {
  apps : [{
    name   : '$proc_name',
    script : '$script',
    interpreter: 'python3',
    min_uptime: '5m',
    max_restarts: '5',
    args: [$joined_args]
  }]
}" > app.config.js

# Print configuration to be used
cat app.config.js

pm2 start app.config.js

# Check if packages are installed.
check_package_installed "jq"
if [ "$?" -eq 1 ]; then
    while true; do

        # First ensure that this is a git installation
        if [ -d "./.git" ]; then

            # check value on github remotely
            latest_version=$(check_variable_value_on_github "opentensor/validators" "openvalidators/__init__.py" "__version__ ")

            # If the file has been updated
            if version_less_than $current_version $latest_version; then
                echo "latest version $latest_version"
                echo "current version $current_version"
                diff=$(get_version_difference $latest_version $current_version)
                if [ "$diff" -eq 1 ]; then
                    echo "current validator version:" "$current_version" 
                    echo "latest validator version:" "$latest_version" 

                    # Pull latest changes
                    # Failed git pull will return a non-zero output
                    if git pull origin $branch; then
                        # latest_version is newer than current_version, should download and reinstall.
                        echo "New version published. Updating the local copy."

                        # Install latest changes just in case.
                        pip install -e .

                        # # Run the Python script with the arguments using pm2
                        # TODO (shib): Remove this pm2 del in the next spec version update.
                        pm2 del auto_run_validator
                        echo "Restarting PM2 process"
                        pm2 restart $proc_name

                        # Update current version:
                        current_version=$(read_version_value)
                        echo ""

                        # Restart autorun script
                        echo "Restarting script..."
                        ./$(basename $0) $old_args && exit
                    else
                        echo "**Will not update**"
                        echo "It appears you have made changes on your local copy. Please stash your changes using git stash."
                    fi
                else
                    # current version is newer than the latest on git. This is likely a local copy, so do nothing. 
                    echo "**Will not update**"
                    echo "The local version is $diff versions behind. Please manually update to the latest version and re-run this script."
                fi
            else
                echo "**Skipping update **"
                echo "$current_version is the same as or more than $latest_version. You are likely running locally."
            fi
        else
            echo "The installation does not appear to be done through Git. Please install from source at https://github.com/opentensor/validators and rerun this script."
        fi
        
        # Wait about 30 minutes
        # This should be plenty of time for validators to catch up
        # and should prevent any rate limitations by GitHub.
        sleep 1800
    done
else
    echo "Missing package 'jq'. Please install it for your system first."
fi
