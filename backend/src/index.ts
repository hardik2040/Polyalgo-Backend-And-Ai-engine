import Fastify from 'fastify';
import cors from '@fastify/cors';
import socketio from 'fastify-socket.io';
import axios from 'axios';
import pino from 'pino';
import { connectDB } from './config/db';
import Prediction from './models/Prediction';
import Trade from './models/Trade';

const logger = pino({ transport: { target: 'pino-pretty' } });
const AI_ENGINE  = process.env.AI_ENGINE_URL ?? 'http://127.0.0.1:8000';
const PORT       = parseInt(process.env.PORT ?? '3000', 10);
const CORS_ORIGIN = (process.env.CORS_ORIGIN ?? '*').split(',').map(s => s.trim());

const fastify = Fastify({
  logger: { transport: { target: 'pino-pretty' } }
});

fastify.register(cors, { origin: CORS_ORIGIN });
fastify.register(socketio, { cors: { origin: CORS_ORIGIN } });

// ═══════════════════════════════════════════════════════════════════════════════
// HEALTH
// ═══════════════════════════════════════════════════════════════════════════════
fastify.get('/api/health', async (req, reply) => {
  const engine = await axios.get(`${AI_ENGINE}/health`).catch(() => ({ data: { status: 'offline' } }));
  return reply.send({ backend: 'ok', ai_engine: engine.data });
});

// ═══════════════════════════════════════════════════════════════════════════════
// BOT CONTROL
// ═══════════════════════════════════════════════════════════════════════════════
fastify.post('/api/bot/start', async (req, reply) => {
  const res = await axios.post(`${AI_ENGINE}/bot/start`).catch(e => ({ data: { error: e.message } }));
  return reply.send(res.data);
});

fastify.post('/api/bot/stop', async (req, reply) => {
  const res = await axios.post(`${AI_ENGINE}/bot/stop`).catch(e => ({ data: { error: e.message } }));
  return reply.send(res.data);
});

fastify.get('/api/bot/status', async (req, reply) => {
  const res = await axios.get(`${AI_ENGINE}/bot/status`).catch(() => ({
    data: { is_running: false, open_positions: 0, total_pnl: 0 }
  }));
  return reply.send({ ...res.data, isBotActive: res.data.is_running });
});

fastify.post('/api/bot/toggle', async (req, reply) => {
  const status = await axios.get(`${AI_ENGINE}/bot/status`).catch(() => ({ data: { is_running: false } }));
  if (status.data.is_running) {
    await axios.post(`${AI_ENGINE}/bot/stop`).catch(() => {});
    return reply.send({ isBotActive: false, action: 'stopped' });
  } else {
    await axios.post(`${AI_ENGINE}/bot/start`).catch(() => {});
    return reply.send({ isBotActive: true, action: 'started' });
  }
});

fastify.post('/api/bot/mode', async (req: any, reply) => {
  const res = await axios.post(`${AI_ENGINE}/bot/mode`, req.body).catch(e => ({ data: { error: e.message } }));
  return reply.send(res.data);
});

fastify.post('/api/bot/settings', async (req: any, reply) => {
  try {
    const res = await axios.post(`${AI_ENGINE}/bot/settings`, req.body);
    return reply.send(res.data);
  } catch (e: any) {
    return reply.status(e.response?.status || 500).send({ error: e.message });
  }
});

// ═══════════════════════════════════════════════════════════════════════════════
// TRADING DATA
// ═══════════════════════════════════════════════════════════════════════════════
fastify.get('/api/trades', async (req, reply) => {
  const botTrades = await axios.get(`${AI_ENGINE}/bot/trades`).catch(() => ({ data: { trades: [] } }));
  const trades = botTrades.data.trades || [];
  const dbTrades = await Trade.find({}).sort({ executedAt: -1 }).limit(20).lean().catch(() => []);
  return reply.send({ success: true, trades: [...trades, ...dbTrades] });
});

fastify.get('/api/signals', async (req, reply) => {
  const res = await axios.get(`${AI_ENGINE}/bot/signals`).catch(() => ({ data: { signals: [] } }));
  return reply.send({ success: true, signals: res.data.signals || [] });
});

fastify.get('/api/positions', async (req, reply) => {
  const res = await axios.get(`${AI_ENGINE}/bot/positions`).catch(() => ({ data: { positions: [] } }));
  return reply.send({ success: true, positions: res.data.positions || [] });
});

fastify.post('/api/positions/:conditionId/sell', async (req: any, reply) => {
  const { conditionId } = req.params;
  const res = await axios.post(`${AI_ENGINE}/positions/${conditionId}/sell`)
    .catch(e => ({ data: { success: false, error: e.message } }));
  return reply.send(res.data);
});

fastify.post('/api/positions/:conditionId/pin', async (req: any, reply) => {
  const { conditionId } = req.params;
  const res = await axios.post(`${AI_ENGINE}/positions/${conditionId}/pin`, req.body)
    .catch(e => ({ data: { success: false, error: e.message } }));
  return reply.send(res.data);
});

// ═══════════════════════════════════════════════════════════════════════════════
// MARKETS
// ═══════════════════════════════════════════════════════════════════════════════
fastify.get('/api/markets', async (req: any, reply) => {
  const weatherOnly = req.query.weather_only === 'true';
  const res = await axios.get(`${AI_ENGINE}/markets/all`, {
    params: { weather_only: weatherOnly, limit: 100 }
  }).catch(() => ({ data: { markets: [] } }));
  return reply.send({ success: true, markets: res.data.markets || [] });
});

fastify.get('/api/markets/weather', async (req, reply) => {
  const res = await axios.get(`${AI_ENGINE}/markets/weather`).catch(() => ({ data: { markets: [] } }));
  return reply.send({ success: true, ...res.data });
});

// ═══════════════════════════════════════════════════════════════════════════════
// AI PREDICTION
// ═══════════════════════════════════════════════════════════════════════════════
fastify.post('/api/predict', async (req: any, reply) => {
  const res = await axios.post(`${AI_ENGINE}/predict`, req.body || {}).catch(e => ({
    data: { error: e.message, probability: 0.5, confidence: 0.1, ev: 0 }
  }));
  return reply.send(res.data);
});

fastify.post('/api/predict-external', async (req: any, reply) => {
  const res = await axios.post(`${AI_ENGINE}/predict-external`, req.body || {}).catch(e => ({
    data: { error: e.message }
  }));
  return reply.send(res.data);
});

fastify.get('/api/weather/predict', async (req: any, reply) => {
  const res = await axios.get(`${AI_ENGINE}/weather/predict`, {
    params: { question: req.query.question || '' }
  }).catch(e => ({ data: { error: e.message, probability: 0.5, confidence: 0.05 } }));
  return reply.send(res.data);
});

// ═══════════════════════════════════════════════════════════════════════════════
// REINFORCEMENT LEARNING
// ═══════════════════════════════════════════════════════════════════════════════
fastify.get('/api/rl/stats', async (req, reply) => {
  const res = await axios.get(`${AI_ENGINE}/rl/stats`).catch(() => ({ data: {} }));
  return reply.send(res.data);
});

fastify.get('/api/rl/qtable', async (req, reply) => {
  const res = await axios.get(`${AI_ENGINE}/rl/qtable`).catch(() => ({ data: {} }));
  return reply.send(res.data);
});

fastify.post('/api/rl/update', async (req: any, reply) => {
  const res = await axios.post(`${AI_ENGINE}/rl/update`, req.body).catch(e => ({
    data: { error: e.message }
  }));
  return reply.send(res.data);
});

// ═══════════════════════════════════════════════════════════════════════════════
// DATA SYNC
// ═══════════════════════════════════════════════════════════════════════════════
fastify.post('/api/sync/trade', async (req: any, reply) => {
  const tradeData = req.body;
  try {
    const trade = await Trade.findOneAndUpdate(
      { tradeId: tradeData.orderId || tradeData.conditionId },
      {
        tradeId:     tradeData.orderId || tradeData.conditionId,
        conditionId: tradeData.conditionId,
        assetId:     tradeData.tokenId || 'UNKNOWN',
        position:    tradeData.side || 'YES',
        stakeAmount: tradeData.stake_usd || 0,
        entryPrice:  tradeData.entryPrice || 0,
        status:      tradeData.status === 'CLOSED' ? 'CLOSED' : 'OPEN',
        pnl:         tradeData.realizedPnl || tradeData.pnl_usd || 0,
        executedAt:  tradeData.enteredAt ? new Date(tradeData.enteredAt) : new Date(),
        closedAt:    tradeData.closedAt ? new Date(tradeData.closedAt) : undefined,
      },
      { upsert: true, new: true }
    );
    return reply.send({ success: true, trade });
  } catch (err: any) {
    return reply.status(500).send({ success: false, error: err.message });
  }
});

fastify.post('/api/sync/prediction', async (req: any, reply) => {
  const predData = req.body;
  try {
    const prediction = new Prediction({
      conditionId:             predData.conditionId,
      modelType:               predData.dataSource || 'orchestrator',
      trueProbability:         predData.trueProbability,
      confidence:              predData.confidence,
      marketProbabilityAtTime: predData.marketProbabilityAtTime,
      expectedValue:           predData.ev,
      externalSignals:         predData.signals || {},
      timestamp:               predData.timestamp ? new Date(predData.timestamp) : new Date(),
    });
    await prediction.save();
    return reply.send({ success: true, prediction });
  } catch (err: any) {
    return reply.status(500).send({ success: false, error: err.message });
  }
});

// ═══════════════════════════════════════════════════════════════════════════════
// PORTFOLIO
// ═══════════════════════════════════════════════════════════════════════════════
fastify.get('/api/portfolio/value', async (req, reply) => {
  const res = await axios.get(`${AI_ENGINE}/portfolio/value`).catch(() => ({
    data: { total_value: 0, positions_count: 0 }
  }));
  return reply.send(res.data);
});

fastify.get('/api/portfolio/positions', async (req, reply) => {
  const res = await axios.get(`${AI_ENGINE}/portfolio/positions`).catch(() => ({
    data: { positions: [] }
  }));
  return reply.send(res.data);
});

fastify.get('/api/portfolio/trades', async (req, reply) => {
  const res = await axios.get(`${AI_ENGINE}/portfolio/trades`).catch(() => ({
    data: { trades: [] }
  }));
  return reply.send(res.data);
});

// ═══════════════════════════════════════════════════════════════════════════════
// WEBSOCKET BROADCAST
// ═══════════════════════════════════════════════════════════════════════════════
const broadcastLiveState = async () => {
  try {
    const [statusR, posR, sigR, pfR, rlR, tradeR] = await Promise.all([
      axios.get(`${AI_ENGINE}/bot/status`).catch(() => ({ data: {} })),
      axios.get(`${AI_ENGINE}/bot/positions`).catch(() => ({ data: { positions: [] } })),
      axios.get(`${AI_ENGINE}/bot/signals`).catch(() => ({ data: { signals: [] } })),
      axios.get(`${AI_ENGINE}/portfolio/value`).catch(() => ({ data: { total_value: 0 } })),
      axios.get(`${AI_ENGINE}/rl/stats`).catch(() => ({ data: {} })),
      axios.get(`${AI_ENGINE}/bot/trades`).catch(() => ({ data: { trades: [] } })),
    ]);

    const state = {
      status:          { ...statusR.data, isBotActive: statusR.data.is_running },
      positions:       posR.data.positions || [],
      signals:         sigR.data.signals || [],
      portfolio_value: pfR.data.total_value || 0,
      rl_stats:        rlR.data,
      trades:          tradeR.data.trades || [],
      timestamp:       new Date().toISOString(),
    };

    fastify.io.emit('state_update', state);
  } catch (err) {
    fastify.log.error('Broadcast error:', err);
  }
};

// ═══════════════════════════════════════════════════════════════════════════════
// STARTUP
// ═══════════════════════════════════════════════════════════════════════════════
const start = async () => {
  try {
    await connectDB();
    await fastify.listen({ port: PORT, host: '0.0.0.0' });
    fastify.log.info(`PolyAlgo Backend running on http://0.0.0.0:${PORT}`);
    setInterval(broadcastLiveState, 3000);
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
};

start();
