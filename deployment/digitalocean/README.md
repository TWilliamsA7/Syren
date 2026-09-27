# DigitalOcean Droplet deployment

This guide deploys the live demo on one Ubuntu Droplet. Nginx serves the built React app and
proxies `/api` to the Python server. A systemd service keeps the API and its child live-feed
process running. Replay is hidden in the public build and its API endpoints are blocked, so the
server does not download multi-gigabyte history archives.

## 1. Create the Droplet

In the DigitalOcean control panel, create an Ubuntu 24.04 Droplet. A 2 GiB shared CPU size is a
reasonable starting point for the live demo. Add an SSH key, enable monitoring, and create a
Cloud Firewall allowing inbound TCP 22 (SSH), 80 (HTTP), and 443 (HTTPS); allow outbound traffic
so the server can reach GitHub, adsb.lol, and Google Gemini. Attach the firewall to the Droplet.

DigitalOcean recommends SSH-key access and a non-root sudo user. Use its
[recommended Droplet setup](https://docs.digitalocean.com/products/droplets/getting-started/recommended-droplet-setup/)
for initial hardening. The app is initially available over HTTP at the Droplet IP. For HTTPS,
point a domain at the IP and follow the optional TLS step below.

Connect from PowerShell, replacing the address with the Droplet's public IPv4 address:

```powershell
ssh root@DROPLET_IP
```

## 2. Install system tools and Python 3.13

Run on the Droplet:

```sh
apt update
apt install -y ca-certificates curl git nginx
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.13
```

Create a dedicated service account, fetch the repository, and prepare its Python environment:

```sh
useradd --system --create-home --home-dir /opt/syren --shell /usr/sbin/nologin syren
git clone YOUR_GIT_REPOSITORY_URL /opt/syren
cd /opt/syren && uv venv --seed --python 3.13 .venv
chown -R syren:syren /opt/syren
```

Replace `YOUR_GIT_REPOSITORY_URL` with the GitHub URL for this project or your fork. The app
does not need the large research dependencies in the root `requirements.txt`. Gemini is
optional; if you want it, install its client in the app environment:

```sh
sudo -u syren /opt/syren/.venv/bin/python -m pip install google-genai
```

## 3. Build the frontend

Install Node.js 22 (which satisfies the repo's Node 22.12+ requirement) using the current
[Node.js installation instructions](https://nodejs.org/en/download/package-manager), then run:

```sh
sudo -u syren -H sh -lc 'cd /opt/syren/frontend && npm ci && VITE_ENABLE_REPLAY=false npm run build'
```

The demo build omits replay controls. The web server config also blocks replay API routes as a
second guard.

## 4. Configure Gemini (optional)

Without a key, live monitoring works and Gemini requests return an explanatory error. To enable
Gemini, create a root-owned environment file and replace the placeholder with your key:

```sh
install -d -m 0750 /etc/syren
sh -c 'umask 077; printf "%s\n" "GEMINI_API_KEY=YOUR_KEY" > /etc/syren/syren.env'
```

The key is read only by the backend process. Nginx limits new Gemini questions to about two
requests per minute per client IP; keep an eye on API usage in Google AI Studio as well.

## 5. Install and start the services

Copy the included service and Nginx configs into place, validate Nginx, and start both services:

```sh
cp /opt/syren/deployment/digitalocean/syren.service /etc/systemd/system/syren.service
cp /opt/syren/deployment/digitalocean/syren-rate-limit.conf /etc/nginx/conf.d/syren-rate-limit.conf
cp /opt/syren/deployment/digitalocean/syren.nginx /etc/nginx/sites-available/syren
ln -s /etc/nginx/sites-available/syren /etc/nginx/sites-enabled/syren
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl daemon-reload
systemctl enable --now syren nginx
```

Open `http://DROPLET_IP` in a browser. The backend binds only to `127.0.0.1:8000`; expose only
Nginx through the Cloud Firewall.

## 6. Check the deployment

On the Droplet:

```sh
systemctl status syren --no-pager
journalctl -u syren -n 50 --no-pager
curl -i http://127.0.0.1:8000/api/aircraft
```

From your computer, open the site and confirm the map updates. The API returns `200` (or `304`
when the feed has not changed). Confirm a replay request returns `404`:

```powershell
Invoke-WebRequest -Method Post -Uri http://DROPLET_IP/api/start_history -ContentType 'application/json' -Body '{"date":"2026-09-25"}'
```

For logs use `journalctl -u syren -f`. To deploy a later revision, run `git pull` as the `syren`
account, rebuild the frontend, then run `systemctl restart syren` (restart Nginx only if its
configuration changed).

## 7. Optional domain and HTTPS

You can leave DNS at your registrar or move DNS management to DigitalOcean. If the domain
already has email or other services, keep its existing DNS provider unless you also copy those
records to DigitalOcean before changing nameservers.

Create DNS records pointing to the Droplet's public IPv4 address:

- Apex/root domain: an `A` record for `@` (some DNS panels call this the blank hostname).
- `www`: a `CNAME` to the apex domain, or an `A` record to the same IPv4 address.
- Add `AAAA` records only if IPv6 is enabled and configured on the Droplet.

If using DigitalOcean DNS, add the domain under **Networking → Domains**, create the records,
then set the registrar's nameservers to the DigitalOcean nameservers shown in its setup guide.
If DNS stays with your registrar, create the records there and do not change nameservers. See
DigitalOcean's [DNS quickstart](https://docs.digitalocean.com/products/networking/dns/getting-started/quickstart/)
and [record guide](https://docs.digitalocean.com/products/networking/dns/how-to/manage-records/).

Wait until the domain resolves to the Droplet IP. Then update the Nginx virtual host and reload:

```sh
sed -i 's/server_name _;/server_name example.com www.example.com;/' /etc/nginx/sites-available/syren
nginx -t && systemctl reload nginx
```

Replace `example.com` with your domain; omit `www.example.com` if you did not create that DNS
name. Keep inbound ports 80 and 443 open in the Cloud Firewall. Install Certbot's Snap package
and let it configure HTTPS for the names you use:

```sh
apt install -y snapd
snap install core
snap refresh core
snap install --classic certbot
ln -s /snap/bin/certbot /usr/local/bin/certbot
certbot --nginx -d example.com -d www.example.com
certbot renew --dry-run
```

Replace the example names in the certificate command with your actual domain names. When
Certbot asks, choose the option to redirect HTTP to HTTPS. Confirm `https://example.com` loads
the app and that `systemctl status certbot.timer` shows the renewal timer. These steps follow
[Certbot's Nginx instructions](https://certbot.eff.org/instructions?os=snap&ws=nginx).
