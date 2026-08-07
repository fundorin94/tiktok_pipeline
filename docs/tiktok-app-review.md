# TikTok app review — checklist for this project

Path chosen: **drafts (inbox) upload**, not Direct Post. Videos land in the account's
TikTok drafts and the owner publishes them from the TikTok app. This needs only the
`video.upload` scope and none of the Direct Post consent UI, which is what makes
review straightforward.

## Before submitting

1. **Host the two policy pages** at public URLs (GitHub Pages, Notion, any static host):
   - `docs/privacy-policy.md`
   - `docs/terms-of-service.md`
   Fill in the name and contact email placeholders first.

2. **In the TikTok developer portal**, on the app:
   - add the **Content Posting API** product, choosing the **upload to drafts** option
     (not Direct Post);
   - scopes: `user.info.basic`, `video.upload`;
   - redirect URI: exactly `https://fundorin94.github.io/tiktok-app-pages/callback.html`
     (TikTok rejects `localhost` — the Pages callback shows the code and you paste
     it into the local app, so the client secret never leaves your machine);
   - paste the two policy URLs.

3. **Add your own TikTok account as a target user / tester** so the sandbox can upload
   to it before approval.

## Recording the demo video

Run the demo app, which exists for exactly this purpose:

```bash
venv/Scripts/python.exe tools/tiktok_demo_app.py
```

Record one continuous screen capture (no cuts) showing:

1. The app's home page, with the explanation of what it does.
2. Clicking **Continue with TikTok** → TikTok's own sign-in and the consent screen
   listing the requested scopes → returning to the app.
3. The connected state showing the account nickname read from `creator_info`.
4. Choosing a rendered video and clicking **Send to TikTok drafts**, then the success
   message with the `publish_id`.
5. Switching to the TikTok app: the upload notification in **Inbox**, opening it, and the
   owner adding a caption and posting it manually.

Step 5 matters most: it demonstrates that publication is a human decision, which is
the core of what review checks for this path.

## What to write in the submission

Describe it plainly and accurately:

> A personal content-production tool. It assembles documentary videos about
> documented criminal cases from public-domain archive photographs and generated
> imagery, then uploads them to the operator's own TikTok account as drafts. It has
> no other users. Nothing is published automatically — the account owner reviews
> every video and posts it from the TikTok app.

Scope justification:
- `user.info.basic` — show which account is connected, so the operator can confirm
  the upload target before sending.
- `video.upload` — place the finished video into that account's drafts.

## After approval

The pipeline posts in drafts mode by default (`TIKTOK_POST_MODE=inbox`). To publish
straight to the feed later you would need the `video.publish` scope and a second
review, plus the Direct Post consent UI: `creator_info` must be queried and its
privacy levels, interaction toggles and commercial-content disclosure shown to the
user before posting.
