# Use the official Python image as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app



# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define the command to run your Streamlit app --theme.base="dark"
ENTRYPOINT streamlit run login.py --server.port=8501 --server.address=0.0.0.0 --theme.base="dark"