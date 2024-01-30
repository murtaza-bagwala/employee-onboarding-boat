import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { CharacterTextSplitter } from "langchain/text_splitter";
import { RetrievalQAChain } from "langchain/chains";
import { OpenAIEmbeddings, OpenAI } from "@langchain/openai";
import { FaissStore } from "@langchain/community/vectorstores/faiss";


export const run = async () => {
    const OPEN_API_KEY = ''
    const loader = new CheerioWebBaseLoader(
    "https://github.com/MethodologyDev/Methodology/"
    );

    const docs = await loader.load();

    console.log(docs.length);
    console.log(docs[0].pageContent.length);

    const splitter = new CharacterTextSplitter({
        separator: "/n",
        chunkSize: 1000,
        chunkOverlap: 200,
    });


    const splitDocs = await splitter.splitDocuments(docs);

    console.log(splitDocs.length);
    console.log(splitDocs[0].pageContent.length);

    const model = new OpenAIEmbeddings({ openAIApiKey: OPEN_API_KEY});

    const vectorStore = await FaissStore.fromDocuments(
        docs,
        model
    );

    // const resultOne = await vectorStore.similaritySearch("hello world", 1);
    // console.log(resultOne);  

    


    const llm = new OpenAI({
        openAIApiKey: OPEN_API_KEY,
        model: "text-davinci-003",
        temperature: 0.9
    });

    const chain = RetrievalQAChain.fromLLM(llm, vectorStore.asRetriever());

    // let res = await chain.call({
    //     query: "Who is Julie",
    // });

    // console.log({ res });

    // res = await chain.call({
    //     query: "Why Julie started methodology",
    // });
    // console.log({ res });

    // res = await chain.call({
    //     query: "who is the cofounder of methodology",
    // });
    // console.log({ res });

    // res = await chain.call({
    //     query: "when methodology was started",
    // });
    // console.log({ res });

    // res = await chain.call({
    //     query: "Why julie replace the intial chef",
    // });
    // console.log({ res });

    let res = await chain.call({
        query: "how to install methodology codebase from github",
    });
    console.log({ res });
}

run()




// urls = ['https://www.gomethodology.com/',
//     'https://www.gomethodology.com/our-story',
//     'https://www.gomethodology.com/blog'
// ]

