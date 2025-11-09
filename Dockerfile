ARG PYTHON_VERSION=3.10

# Use a base image
FROM python:${PYTHON_VERSION}

# Define the build argument and set a default value
ARG MODULES=requirements.txt

# Set the environment variable from the argument
ENV MODULES=${MODULES}

# Add main script
COPY requirements.txt .

# If MODULES is provided as a list, handle the installation
# Otherwise, treat MODULES as a file
RUN if [ -f "$MODULES" ]; then \
    # Copy the default file if MODULES is a file
    echo "Using default file: $MODULES"; \
    pip install -r "$MODULES"; \
else \
    echo "Using list of modules: $MODULES"; \
    pip install $MODULES; \
fi

ENTRYPOINT bash
WORKDIR /code