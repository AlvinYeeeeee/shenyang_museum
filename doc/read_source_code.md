
## Here is the full call stack
```bash
_chat (/home/xiachunwei/Software/anaconda3/envs/glm/lib/python3.11/site-packages/llama_index/llms/openai/base.py:453)
__call__ (/home/xiachunwei/Software/anaconda3/envs/glm/lib/python3.11/site-packages/tenacity/__init__.py:478)
wrapped_f (/home/xiachunwei/Software/anaconda3/envs/glm/lib/python3.11/site-packages/tenacity/__init__.py:336)
wrapper (/home/xiachunwei/Software/anaconda3/envs/glm/lib/python3.11/site-packages/llama_index/llms/openai/base.py:106)
chat (/home/xiachunwei/Software/anaconda3/envs/glm/lib/python3.11/site-packages/llama_index/llms/openai/base.py:355)
wrapped_llm_chat (/home/xiachunwei/Software/anaconda3/envs/glm/lib/python3.11/site-packages/llama_index/core/llms/callbacks.py:173)
wrapper (/home/xiachunwei/Software/anaconda3/envs/glm/lib/python3.11/site-packages/llama_index/core/instrumentation/dispatcher.py:311)
predict (/home/xiachunwei/Software/anaconda3/envs/glm/lib/python3.11/site-packages/llama_index/core/llms/llm.py:596)
wrapper (/home/xiachunwei/Software/anaconda3/envs/glm/lib/python3.11/site-packages/llama_index/core/instrumentation/dispatcher.py:311)
__call__ (/home/xiachunwei/Software/anaconda3/envs/glm/lib/python3.11/site-packages/llama_index/core/response_synthesizers/refine.py:85)
wrapper (/home/xiachunwei/Software/anaconda3/envs/glm/lib/python3.11/site-packages/llama_index/core/instrumentation/dispatcher.py:311)
_give_response_single (/home/xiachunwei/Software/anaconda3/envs/glm/lib/python3.11/site-packages/llama_index/core/response_synthesizers/refine.py:241)
get_response (/home/xiachunwei/Software/anaconda3/envs/glm/lib/python3.11/site-packages/llama_index/core/response_synthesizers/refine.py:179)
wrapper (/home/xiachunwei/Software/anaconda3/envs/glm/lib/python3.11/site-packages/llama_index/core/instrumentation/dispatcher.py:311)
get_response (/home/xiachunwei/Software/anaconda3/envs/glm/lib/python3.11/site-packages/llama_index/core/response_synthesizers/compact_and_refine.py:43)
wrapper (/home/xiachunwei/Software/anaconda3/envs/glm/lib/python3.11/site-packages/llama_index/core/instrumentation/dispatcher.py:311)
synthesize (/home/xiachunwei/Software/anaconda3/envs/glm/lib/python3.11/site-packages/llama_index/core/response_synthesizers/base.py:241)
wrapper (/home/xiachunwei/Software/anaconda3/envs/glm/lib/python3.11/site-packages/llama_index/core/instrumentation/dispatcher.py:311)
_query (/home/xiachunwei/Software/anaconda3/envs/glm/lib/python3.11/site-packages/llama_index/core/query_engine/retriever_query_engine.py:179)
wrapper (/home/xiachunwei/Software/anaconda3/envs/glm/lib/python3.11/site-packages/llama_index/core/instrumentation/dispatcher.py:311)
query (/home/xiachunwei/Software/anaconda3/envs/glm/lib/python3.11/site-packages/llama_index/core/base/base_query_engine.py:52)
wrapper (/home/xiachunwei/Software/anaconda3/envs/glm/lib/python3.11/site-packages/llama_index/core/instrumentation/dispatcher.py:311)
query (/home/xiachunwei/Projects/rag-it/src/rag_llama_index.py:25)
<module> (/home/xiachunwei/Projects/rag-it/src/test_llama_index.py:14)
```

## Here is the example message

[{'role': 'system', 'content': "You are an expert Q&A system that is trusted around the world.\nAlways answer the query using the provided context information, and not prior knowledge.\nSome rules to follow:\n1. Never directly reference the given context in your answer.\n2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines."}, 
{'role': 'user', 'content': 'Context information is below.\n---------------------\nfile_path: /home/xiachunwei/Projects/rag-it/data/text/shenyang.txt\n\n有专家推断它有可能是氏族首领使用的具有图腾意义的发簪，也许只有首领才能佩戴。新乐先民们将光明当成信仰，幻化成鹏鸟，创造出令人瞩目的新乐文化，成为沈阳文明的第一缕曙光。"\n\n偏堡子文化0004\t5000——4000年前，沈阳地区较为活跃的文化叫作偏堡子文化，它为沈阳历史迈入青铜时代准备和创造了条件。\n\n青铜时代0005\t距今约3300年，先后经历了高台山文化和新乐上层文化，从出土大量的随葬品中发现了一个特殊的现象，几乎每个墓葬中都会有壶和钵，钵是倒扣在壶口上的。这一独特的葬俗也成为高台山文化的典型特点。新乐上层文化出现的时代略晚于高台山文化，是沈阳地区比较成熟的青铜文化。这个时期出现了大量用于蒸煮的鼎、甗、鬲等陶器炊具。这也说明早在三四千年前的青铜时代，新乐先民的饮食已经有了很大的改善，由肉食、野果为主改为以粮食为主了。\n\n郑家洼子文化0006\t而在这一时期最具代表性的是郑家洼子青铜短剑文化，这是1965年在沈阳西南的郑家洼子遗址挖掘出土的一座墓葬，后经专家认定是目前所见东北亚地区出土的青铜短剑墓葬中规模最大、随葬品最多的一座。墓的年代大致在春秋后期，墓主人是一位五六十岁的男性。墓葬中出土的曲刃青铜短剑极具特点，两侧叶刃呈波浪弧线形曲刃，中间有柱状脊与短茎相贯连，可以看出主人有一定的政治地位。在他魂归天际后，后人将他生前最爱的青铜短剑放入墓中伴他长眠。将绿松石串珠戴在他的身上，2000多年前的沈阳并非繁盛之地，更不出产绿松石等特色资源，但这位长居于此的墓主人竟能够有这样的饰品，也足见其地位的显赫和生活的奢靡。\n\n汉唐一统，秦汉开拓0007\t战国后期，随着燕国不断强大，开始向东北地区开疆辟土。这是燕国大将秦开，当年他却东胡、建城邑、筑长城、设五郡，被正式纳入到中原王朝燕国统治的版图中，也由此进入到了一个新的历史阶段。\n\nfile_path: /home/xiachunwei/Projects/rag-it/data/text/shenyang.txt\n\n这是在中小贵族的墓葬中发现的铜鎏金面具，从出土的样式来看，它的确没有高等级面具那样富有神采。这是在中小贵族的墓葬中发现的辽三彩。辽三彩是独具游牧民族风格的低温釉陶瓷器。辽三彩虽质量不如唐三彩，但也有自己鲜明的特色。除了受契丹民族传统文化影响外，还常常可以看到中原文化、西亚文化的影子，这无疑是多元文化强烈影响的结果。\n\n辽代汉人、渤海遗民0016\t\u200b除了推行因俗而治，辽代特有的官制也是极具特点。在皇帝之下设立两套机构，称南北面官制。分别管理汉人和契丹人。\n\n金代墓葬0017\t辽金时期，随着民族融合的深入，葬俗文化影响也极为深远。近年来在沈阳地区发现了一批金代墓葬，这件镶嵌绿松石的金牌饰就是金代墓葬中出土的一件为数不多的精品。它既有淳朴自然的生活气息，同时又有浓厚的地方民族特色。\n\n畜牧渔猎、农耕稼墙、百业兴盛0018\t"畜牧和渔猎是东北草原民族代代传承的模式，他们在长期的生活中积累了丰富的生产经验。这些铁马镫都是当年跟随契丹人、女真人、渤海人“逐水草而游牧”的见证。沈阳地区当时已经具备了成熟的铁器生产技术，为农耕提供了先进的工具，也成为农业生产力的保障。随着经济的发展，中原宋朝铸造的铜钱大量进入，辽虽也有自铸的钱币，但数量少，宋钱成为辽代主要的流通货币。"\n\n文艺同辉0019\t"是经济繁荣，商业兴盛，推动了文化艺术的发展。辽金时期各类雕塑异彩纷呈。法库叶茂台辽墓出土了两幅珍贵的绢画。一幅是山水画《山弈候约图》，一幅是花鸟画《竹雀双兔图》。《山弈候约图》描绘的是主人设下棋局，等待宾客赴约对弈的情景。这是从墓中出土的21颗黑曜石围棋棋子，我们可以想象，主人是多么痴迷下棋，才会把它带入墓中继续与自己为伴。"\n\n宗教信仰0020\t"沈阳民间至今仍流传着“辽修塔，清修庙”的说法。辽沈地区寺院广布，佛塔林立。\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: 请介绍一下郑家洼子文化\nAnswer: '}]
