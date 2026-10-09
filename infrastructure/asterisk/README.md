# Asterisk voice gateway

Local telephony adapter for UniVoice AI.

## Current scope

- Asterisk in Docker
- PJSIP
- Local SIP endpoint 1000
- DTMF menu: 1, 2, 0
- RTP ports 10000-10100

- GSM gateway with a SIM card: `local/pjsip_gsm.conf.example`,
  see docs/local-run.md

No SIP provider trunk is configured yet.

## Local test

Start the stack:

```bash
docker compose up --build
```

Register a SIP softphone:

- server: `<host-ip>:5060`
- transport: UDP
- username: `1000`
- password: `change-me`

Call extension `1000`.

The sample Asterisk sounds are temporary placeholders. Uzbek prompts will be added with the TTS/audio layer.

The SIP password is development-only and must be replaced before deployment.
