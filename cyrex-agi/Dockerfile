FROM python:3.11-slim

WORKDIR /app


# Bedd runtime (Bun-style) — glibc binary
ARG BEDD_IMAGE=ghcr.io/team-deepiri/bedd:0.6
COPY --from=${BEDD_IMAGE} /usr/local/bin/bedd /usr/local/bin/bedd
COPY --from=${BEDD_IMAGE} /opt/bedd/skills /opt/bedd/skills
ENV BEDD_SKILLS_DIR=/opt/bedd/skills


# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8003

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
