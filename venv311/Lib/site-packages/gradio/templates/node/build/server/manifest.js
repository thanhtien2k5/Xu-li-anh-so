const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "_app",
	assets: new Set([]),
	mimeTypes: {},
	_: {
		client: {start:"_app/immutable/entry/start.ibAPkqOh.js",app:"_app/immutable/entry/app.BPzbTece.js",imports:["_app/immutable/entry/start.ibAPkqOh.js","_app/immutable/chunks/Cd9ZpB24.js","_app/immutable/chunks/BiJ6z1av.js","_app/immutable/chunks/DcUPjU6e.js","_app/immutable/entry/app.BPzbTece.js","_app/immutable/chunks/BScMEiXW.js","_app/immutable/chunks/BiJ6z1av.js","_app/immutable/chunks/DcUPjU6e.js","_app/immutable/chunks/D3Ta_w6-.js","_app/immutable/chunks/CKiwnvWA.js"],stylesheets:[],fonts:[],uses_env_dynamic_public:false},
		nodes: [
			__memo(() => import('./chunks/0-iT5OTOYR.js')),
			__memo(() => import('./chunks/1-jQb6M_Vl.js')),
			__memo(() => import('./chunks/2-C5G_QOs4.js').then(function (n) { return n._; }))
		],
		remotes: {
			
		},
		routes: [
			{
				id: "/[...catchall]",
				pattern: /^(?:\/([^]*))?\/?$/,
				params: [{"name":"catchall","optional":false,"rest":true,"chained":true}],
				page: { layouts: [0,], errors: [1,], leaf: 2 },
				endpoint: null
			}
		],
		prerendered_routes: new Set([]),
		matchers: async () => {
			
			return {  };
		},
		server_assets: {}
	}
}
})();

const prerendered = new Set([]);

const base = "";

export { base, manifest, prerendered };
//# sourceMappingURL=manifest.js.map
