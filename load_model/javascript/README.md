## Install a tool to run a http server

For exemple: `light-server`

```bash
npm install light-server
```

## Serve the current folder

```bash
npx light-server -s . -p 8080
```

## Open your browser

Open your browser and go to http://localhost:8080.

Expected output

```
URL: https://en.wikipedia.org/wiki/Phishing
Likelihood of being a phishing site: 0.00%
----
URL: http//weird-website.com
Likelihood of being a phishing site: 66.34%
----
```
