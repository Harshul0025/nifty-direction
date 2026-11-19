// Vercel Cron calls this every weekday morning and afternoon (times are in vercel.json).
// It starts the GitHub Actions workflow "Daily prediction", which makes the prediction
// and saves it in site/data/live.json. GitHub's own schedule often runs hours late,
// so we start the workflow from here instead.
module.exports = async (req, res) => {
  // only Vercel Cron knows this secret, so nobody else can start the workflow
  if (req.headers.authorization !== `Bearer ${process.env.CRON_SECRET}`) {
    return res.status(401).send("not allowed");
  }
  const reply = await fetch(
    "https://api.github.com/repos/Harshul0025/nifty-direction/actions/workflows/daily.yml/dispatches",
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.GH_TOKEN}`,
        Accept: "application/vnd.github+json",
        "User-Agent": "nifty-direction-cron",
      },
      body: JSON.stringify({ ref: "main" }),
    }
  );
  // GitHub answers 204 when the workflow has started
  res.status(reply.status === 204 ? 200 : 500).send(`GitHub answered ${reply.status}`);
};
