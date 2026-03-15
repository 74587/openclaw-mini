/**
 * Gateway 模块导出
 */

export { startGatewayServer, type GatewayServer, type GatewayServerOptions } from "./server.js";
export { GatewayClient, type GatewayClientOptions } from "./client.js";
export {
  type RequestFrame, type ResponseFrame, type EventFrame, type HelloOk,
  type ErrorShape, type GatewayFrame,
  ErrorCodes, errorShape,
  PROTOCOL_VERSION, GATEWAY_METHODS, GATEWAY_EVENTS,
} from "./protocol.js";
export { handlers, type Handler, type HandlerContext } from "./handlers.js";
