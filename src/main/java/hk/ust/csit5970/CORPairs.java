package hk.ust.csit5970;

import org.apache.commons.cli.*;
import org.apache.commons.cli.Options;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;


import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.*;
import java.util.Arrays;
import java.util.HashMap;

/**
 * Compute the bigram count using "pairs" approach
 */
public class CORPairs extends Configured implements Tool {
	private static final Logger LOG = Logger.getLogger(CORPairs.class);

	/*
	 * First-pass Mapper
	 */
	private static class CORMapper1 extends
			Mapper<LongWritable, Text, Text, IntWritable> {
		private Text wordText = new Text();
		
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			// Please use this tokenizer! DO NOT implement a tokenizer by yourself!
			String clean_doc = value.toString().replaceAll("[^a-zA-Z ]", " ");
            StringTokenizer doc_tokenizer = new StringTokenizer(clean_doc);
            
            // Use a map to combine counts for each word in this line.
            Map<String, Integer> wordCountMap = new HashMap<String, Integer>();
            while (doc_tokenizer.hasMoreTokens()) {
                String token = doc_tokenizer.nextToken();
                if (!token.isEmpty()) {
                    // Manually update the count without using getOrDefault.
                    if (wordCountMap.containsKey(token)) {
                        wordCountMap.put(token, wordCountMap.get(token) + 1);
                    } else {
                        wordCountMap.put(token, 1);
                    }
                }
            }
            // Emit each word and its count in this line.
            for (Map.Entry<String, Integer> entry : wordCountMap.entrySet()) {
                wordText.set(entry.getKey());
                context.write(wordText, new IntWritable(entry.getValue()));
            }
		}
	}

	/*
	 * First-pass reducer
	 */
	private static class CORReducer1 extends
			Reducer<Text, IntWritable, Text, IntWritable> {
		private static final IntWritable SUM = new IntWritable();
		
		@Override
		public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            context.write(key, new IntWritable(sum));
		}
	}


	/*
	 * Second-pass Mapper
	 */
	public static class CORPairsMapper2 extends Mapper<LongWritable, Text, PairOfStrings, IntWritable> {
		private final static IntWritable one = new IntWritable(1);
		private static final PairOfStrings pair = new PairOfStrings();
    
		@Override
		protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			String line = value.toString().replaceAll("[^a-zA-Z ]", " ");
            StringTokenizer tokenizer = new StringTokenizer(line);
            Set<String> uniqueWords = new HashSet<String>();
            while (tokenizer.hasMoreTokens()) {
                String token = tokenizer.nextToken();
                if (!token.isEmpty()) {
                    uniqueWords.add(token);
                }
            }
            // Convert the set to a sorted list so that A is alphabetically smaller than B.
            List<String> wordList = new ArrayList<String>(uniqueWords);
            Collections.sort(wordList);
            int n = wordList.size();
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    // Create a pair with the first element lexicographically smaller.
                    PairOfStrings pair = new PairOfStrings(wordList.get(i), wordList.get(j));
                    context.write(pair, one);
                }
            }
		}
	}

	/*
	 * Second-pass Combiner
	 */
	private static class CORPairsCombiner2 extends Reducer<PairOfStrings, IntWritable, PairOfStrings, IntWritable> {
		private static final IntWritable SUM = new IntWritable();
		
		@Override
		protected void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
			int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            context.write(key, new IntWritable(sum));
		}
	}

	/*
	 * Second-pass Reducer
	 */
	public static class CORPairsReducer2 extends Reducer<PairOfStrings, IntWritable, PairOfStrings, DoubleWritable> {
		private final static Map<String, Integer> word_total_map = new HashMap<String, Integer>();

		/*
		 * Preload the middle result file.
		 * In the middle result file, each line contains a word and its frequency Freq(A), seperated by "\t"
		 */
		@Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Path middle_result_path = new Path("mid/part-r-00000");
            Configuration middle_conf = new Configuration();
            try {
                FileSystem fs = FileSystem.get(URI.create(middle_result_path.toString()), middle_conf);
                if (!fs.exists(middle_result_path)) {
                    throw new IOException(middle_result_path.toString() + " not exist!");
                }
                FSDataInputStream in = fs.open(middle_result_path);
                BufferedReader reader = new BufferedReader(new InputStreamReader(in));
                LOG.info("Reading middle result file...");
                String line = reader.readLine();
                while (line != null) {
                    String[] parts = line.split("\t");
                    if (parts.length == 2) {
                        word_total_map.put(parts[0], Integer.valueOf(parts[1]));
                    }
                    line = reader.readLine();
                }
                reader.close();
                LOG.info("Finished reading middle result file.");
            } catch (Exception e) {
                System.out.println(e.getMessage());
            }
        }

		/*
		 * Second-pass Reducer
		 */
		@Override
		protected void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
			// Calculate Freq(A,B) - the number of lines containing both words
            int pairFreq = 0;
            for (IntWritable val : values) {
                pairFreq += val.get();
            }
            String wordA = key.getLeftElement();
            String wordB = key.getRightElement();
            Integer freqA = word_total_map.get(wordA);
            Integer freqB = word_total_map.get(wordB);
            if (freqA != null && freqB != null && freqA != 0 && freqB != 0) {
                double cor = (double) pairFreq / (freqA * freqB);
                context.write(key, new DoubleWritable(cor));
            }
		}
	}

	private static final class MyPartitioner extends Partitioner<PairOfStrings, FloatWritable> {
		@Override
		public int getPartition(PairOfStrings key, FloatWritable value, int numReduceTasks) {
			return (key.getLeftElement().hashCode() & Integer.MAX_VALUE) % numReduceTasks;
		}
	}

	/**
	 * Creates an instance of this tool.
	 */
	public CORPairs() {
	}

	private static final String INPUT = "input";
	private static final String MIDDLE = "middle";
	private static final String OUTPUT = "output";
	private static final String NUM_REDUCERS = "numReducers";

	/**
	 * Runs this tool.
	 */
	@SuppressWarnings({ "static-access" })
	public int run(String[] args) throws Exception {
		Options options = new Options();

		options.addOption(OptionBuilder.withArgName("path").hasArg()
				.withDescription("input path").create(INPUT));
		options.addOption(OptionBuilder.withArgName("path").hasArg()
				.withDescription("output path").create(OUTPUT));
		options.addOption(OptionBuilder.withArgName("num").hasArg()
				.withDescription("number of reducers").create(NUM_REDUCERS));

		CommandLine cmdline;
		CommandLineParser parser = new GnuParser();

		try {
			cmdline = parser.parse(options, args);
		} catch (ParseException exp) {
			System.err.println("Error parsing command line: "
					+ exp.getMessage());
			return -1;
		}

		// Lack of arguments
		if (!cmdline.hasOption(INPUT) || !cmdline.hasOption(OUTPUT)) {
			System.out.println("args: " + Arrays.toString(args));
			HelpFormatter formatter = new HelpFormatter();
			formatter.setWidth(120);
			formatter.printHelp(this.getClass().getName(), options);
			ToolRunner.printGenericCommandUsage(System.out);
			return -1;
		}

		String inputPath = cmdline.getOptionValue(INPUT);
		String middlePath = "mid";
		String outputPath = cmdline.getOptionValue(OUTPUT);

		int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ? Integer
				.parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

		LOG.info("Tool: " + CORPairs.class.getSimpleName());
		LOG.info(" - input path: " + inputPath);
		LOG.info(" - output path: " + outputPath);
		LOG.info(" - number of reducers: " + reduceTasks);

		// Setup for the first-pass MapReduce
		Configuration conf1 = new Configuration();

		Job job1 = Job.getInstance(conf1, "Firstpass");

		job1.setJarByClass(CORPairs.class);
		job1.setMapperClass(CORMapper1.class);
		job1.setReducerClass(CORReducer1.class);
		job1.setOutputKeyClass(Text.class);
		job1.setOutputValueClass(IntWritable.class);

		FileInputFormat.setInputPaths(job1, new Path(inputPath));
		FileOutputFormat.setOutputPath(job1, new Path(middlePath));

		// Delete the output directory if it exists already.
		Path middleDir = new Path(middlePath);
		FileSystem.get(conf1).delete(middleDir, true);

		// Time the program
		long startTime = System.currentTimeMillis();
		job1.waitForCompletion(true);
		LOG.info("Job 1 Finished in " + (System.currentTimeMillis() - startTime)
				/ 1000.0 + " seconds");

		// Setup for the second-pass MapReduce

		// Delete the output directory if it exists already.
		Path outputDir = new Path(outputPath);
		FileSystem.get(conf1).delete(outputDir, true);


		Configuration conf2 = new Configuration();
		Job job2 = Job.getInstance(conf2, "Secondpass");

		job2.setJarByClass(CORPairs.class);
		job2.setMapperClass(CORPairsMapper2.class);
		job2.setCombinerClass(CORPairsCombiner2.class);
		job2.setReducerClass(CORPairsReducer2.class);

		job2.setOutputKeyClass(PairOfStrings.class);
		job2.setOutputValueClass(DoubleWritable.class);
		job2.setMapOutputValueClass(IntWritable.class);
		job2.setNumReduceTasks(reduceTasks);

		FileInputFormat.setInputPaths(job2, new Path(inputPath));
		FileOutputFormat.setOutputPath(job2, new Path(outputPath));

		// Time the program
		startTime = System.currentTimeMillis();
		job2.waitForCompletion(true);
		LOG.info("Job 2 Finished in " + (System.currentTimeMillis() - startTime)
				/ 1000.0 + " seconds");

		return 0;
	}

	/**
	 * Dispatches command-line arguments to the tool via the {@code ToolRunner}.
	 */
	public static void main(String[] args) throws Exception {
		ToolRunner.run(new CORPairs(), args);
	}
}