FROM node:24.22.0-bookworm-slim AS deps
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund

FROM node:24.22.0-bookworm-slim AS build
WORKDIR /web
# Next.js bakes rewrite destinations into the build, so the API address is a build argument.
ARG API_INTERNAL_URL=http://api:8000
ENV NEXT_TELEMETRY_DISABLED=1 API_INTERNAL_URL=${API_INTERNAL_URL}
COPY --from=deps /web/node_modules ./node_modules
COPY web/ ./
RUN npm run build

FROM node:24.22.0-bookworm-slim
WORKDIR /web
ARG API_INTERNAL_URL=http://api:8000
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 PORT=3000 HOSTNAME=0.0.0.0 \
    API_INTERNAL_URL=${API_INTERNAL_URL}
COPY --from=build --chown=node:node /web/.next/standalone ./
COPY --from=build --chown=node:node /web/.next/static ./.next/static
COPY --from=build --chown=node:node /web/public ./public
USER node
EXPOSE 3000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:3000/').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"
CMD ["node", "server.js"]
