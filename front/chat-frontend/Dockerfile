# Build stage
FROM node AS builder

# Set working directory inside the container
USER node
WORKDIR /app
# Copy package files and install dependencies
COPY --chown=node:node package*.json ./
RUN npm ci

# Copy the rest of the application code
COPY --chown=node:node . .

# Build the React app
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy the build output to Nginx's html directory
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Start Nginx server
CMD ["nginx", "-g", "daemon off;"]
