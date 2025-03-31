package hk.ust.csit5970;

import java.io.IOException;
import java.util.Arrays;
import java.util.Map;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

/**
 * Compute the bigram counts using the "stripes" approach and then compute
 * relative frequencies. For each word A, the program outputs a record for the total count
 * (the single word frequency) and then one record per succeeding word B with the relative frequency P(B|A).
 * The output format (tab-delimited) is as follows:
 *
 * A   *       totalCount
 * A   B       relativeFreq
 *
 * For example:
 * It	*	2.0
 * It	was	1.0
 * best	*	1.0
 * best	of	1.0
 * ...
 */
public class BigramFrequencyStripes extends Configured implements Tool {
    private static final Logger LOG = Logger.getLogger(BigramFrequencyStripes.class);

    /*
     * Mapper: emits <word, stripe> where stripe is a hash map (of type
     * HashMapStringIntWritable)
     */
    private static class MyMapper extends Mapper<LongWritable, Text, Text, HashMapStringIntWritable> {

        @Override
        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String line = value.toString();
            String[] words = line.trim().split("\\s+");
            if (words.length < 2)
                return;

            // For each word (except the last one), emit a new stripe for the next word.
            for (int i = 0; i < words.length - 1; i++) {
                if (words[i].length() == 0)
                    continue;

                Text wordKey = new Text(words[i]);
                // Create a new stripe for each bigram output.
                HashMapStringIntWritable stripe = new HashMapStringIntWritable();
                stripe.increment(words[i + 1]);
                context.write(wordKey, stripe);
            }
        }
    }

    /*
     * Reducer: aggregates all stripes associated with each key, computes the total count,
     * and then emits the total count (single word frequency) and the relative frequencies.
     */
    private static class MyReducer extends Reducer<Text, HashMapStringIntWritable, PairOfStrings, FloatWritable> {

        @Override
        public void reduce(Text key, Iterable<HashMapStringIntWritable> stripes, Context context)
                throws IOException, InterruptedException {
            // Use a local stripe to sum counts.
            HashMapStringIntWritable sumStripe = new HashMapStringIntWritable();

            // Combine all stripes for the current key.
            for (HashMapStringIntWritable stripe : stripes) {
                for (String nextWord : stripe.keySet()) {
                    sumStripe.increment(nextWord, stripe.get(nextWord));
                }
            }

            // Calculate the total count for the key.
            float totalCount = 0.0f;
            for (String nextWord : sumStripe.keySet()) {
                totalCount += sumStripe.get(nextWord);
            }

            // Emit the single word frequency record.
            // Here, we use "*" as a marker in the second field to indicate the total count.
            PairOfStrings totalRecord = new PairOfStrings(key.toString(), "*");
            context.write(totalRecord, new FloatWritable(totalCount));

            // Emit relative frequencies for each succeeding word.
            for (String nextWord : sumStripe.keySet()) {
                PairOfStrings bigramRecord = new PairOfStrings(key.toString(), nextWord);
                float relativeFreq = sumStripe.get(nextWord) / totalCount;
                context.write(bigramRecord, new FloatWritable(relativeFreq));
            }
        }
    }

    /*
     * Combiner: aggregates all stripes with the same key.
     * We create a new stripe for each output to avoid mutable object reuse.
     */
    private static class MyCombiner extends Reducer<Text, HashMapStringIntWritable, Text, HashMapStringIntWritable> {
        @Override
        public void reduce(Text key, Iterable<HashMapStringIntWritable> stripes, Context context)
                throws IOException, InterruptedException {
            HashMapStringIntWritable sumStripe = new HashMapStringIntWritable();

            // Combine all stripes for the current key.
            for (HashMapStringIntWritable stripe : stripes) {
                for (String nextWord : stripe.keySet()) {
                    sumStripe.increment(nextWord, stripe.get(nextWord));
                }
            }

            // Create a new stripe object to safely emit.
            HashMapStringIntWritable outStripe = new HashMapStringIntWritable();
            for (String w : sumStripe.keySet()) {
                outStripe.put(w, sumStripe.get(w));
            }
            context.write(key, outStripe);
        }
    }

    /**
     * Creates an instance of this tool.
     */
    public BigramFrequencyStripes() {
    }

    private static final String INPUT = "input";
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
            System.err.println("Error parsing command line: " + exp.getMessage());
            return -1;
        }

        // Lack of arguments.
        if (!cmdline.hasOption(INPUT) || !cmdline.hasOption(OUTPUT)) {
            System.out.println("args: " + Arrays.toString(args));
            HelpFormatter formatter = new HelpFormatter();
            formatter.setWidth(120);
            formatter.printHelp(this.getClass().getName(), options);
            ToolRunner.printGenericCommandUsage(System.out);
            return -1;
        }

        String inputPath = cmdline.getOptionValue(INPUT);
        String outputPath = cmdline.getOptionValue(OUTPUT);
        int reduceTasks = cmdline.hasOption(NUM_REDUCERS)
                ? Integer.parseInt(cmdline.getOptionValue(NUM_REDUCERS))
                : 1;

        LOG.info("Tool: " + BigramFrequencyStripes.class.getSimpleName());
        LOG.info(" - input path: " + inputPath);
        LOG.info(" - output path: " + outputPath);
        LOG.info(" - number of reducers: " + reduceTasks);

        // Create and configure a MapReduce job.
        Configuration conf = getConf();
        Job job = Job.getInstance(conf);
        job.setJobName(BigramFrequencyStripes.class.getSimpleName());
        job.setJarByClass(BigramFrequencyStripes.class);

        job.setNumReduceTasks(reduceTasks);

        FileInputFormat.setInputPaths(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(HashMapStringIntWritable.class);
        job.setOutputKeyClass(PairOfStrings.class);
        job.setOutputValueClass(FloatWritable.class);

        /*
         * A MapReduce program consists of four components: a mapper, a reducer,
         * an optional combiner, and an optional partitioner.
         */
        job.setMapperClass(MyMapper.class);
        job.setCombinerClass(MyCombiner.class);
        job.setReducerClass(MyReducer.class);

        // Delete the output directory if it exists already.
        Path outputDir = new Path(outputPath);
        FileSystem.get(conf).delete(outputDir, true);

        // Time the program.
        long startTime = System.currentTimeMillis();
        job.waitForCompletion(true);
        LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

        return 0;
    }

    /**
     * Dispatches command-line arguments to the tool via the {@code ToolRunner}.
     */
    public static void main(String[] args) throws Exception {
        ToolRunner.run(new BigramFrequencyStripes(), args);
    }
}
